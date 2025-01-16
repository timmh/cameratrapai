# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multiprocessing utilities to run SpeciesNet.

Provides utilities for running the SpeciesNet model with various forms
of parallelization, including multi-threading and multiprocessing. It
implements a main `SpeciesNet` class, that serves as a high level 
interface to interact with the different SpeciesNet components using 
different parallelization strategies.
"""
__all__ = [
    "SpeciesNet",
]

import multiprocessing as mp
from multiprocessing.managers import SyncManager
from multiprocessing.pool import ThreadPool
from pathlib import Path
import queue
import threading
import traceback
from typing import Any, Callable, Literal, Optional, Union

from absl import logging
import PIL
import PIL.Image
from tqdm import tqdm

from speciesnet.classifier import SpeciesNetClassifier
from speciesnet.detector import SpeciesNetDetector
from speciesnet.ensemble import SpeciesNetEnsemble
from speciesnet.geolocation import find_admin1_region
from speciesnet.utils import BBox
from speciesnet.utils import load_partial_predictions
from speciesnet.utils import load_rgb_image
from speciesnet.utils import prepare_instances_dict
from speciesnet.utils import PreprocessedImage
from speciesnet.utils import save_predictions

# Handy type aliases.
StrPath = Union[str, Path]
DetectorInput = tuple[str, Optional[PreprocessedImage]]
BBoxOutput = tuple[str, list[BBox]]
ClassifierInput = tuple[str, Optional[PreprocessedImage]]

# Register SpeciesNet model components with the SyncManager to be able to
#  safely share them between processes.
SyncManager.register("Classifier", SpeciesNetClassifier)
SyncManager.register("Detector", SpeciesNetDetector)
SyncManager.register("Ensemble", SpeciesNetEnsemble)


class RepeatedAction(threading.Thread):
    """Repeated action to run regularly at a given interval.

    Implements a threading mechanism to execute a specific function
    repeatedly at set time intervals. It's commonly used for background
    tasks such as saving partial results periodically during long-running
    inference jobs.
    """

    def __init__(self, interval: float, fn: Callable, *fn_args, **fn_kwargs) -> None:
        """Initializes the repeated action.

        Args:
            interval:
                Number of seconds (can be fractional) to wait before repeating the
                action.
            fn:
                Callable representing the action to repeat.
            *fn_args:
                Arguments for the action to repeat.
            **fn_kwargs:
                Keyword arguments for the action to repeat.
        """

        super().__init__()
        self._interval = interval
        self._fn = fn
        self._fn_args = fn_args
        self._fn_kwargs = fn_kwargs
        self._stopped = threading.Event()

    def run(self) -> None:
        """Starts the repeated action."""

        while not self._stopped.is_set():
            self._stopped.wait(self._interval)
            if self._stopped.is_set():
                break
            self._fn(*self._fn_args, **self._fn_kwargs)

    def stop(self) -> None:
        """Stops the repeated action."""

        self._stopped.set()


class Progress:
    """Progress tracker for different components of SpeciesNet.

    Provides a mechanism to track the progress of various tasks within 
    the SpeciesNet inference process. It uses `tqdm` progress bars to
    visually show the status of each component, like detector preprocessing,
    detector prediction, classifier preprocessing, classifier prediction, and
    geolocation operations. It offers a way to update individual trackers as the
    inference progresses, and to stop tracking when inference is complete.
    """

    def __init__(
        self,
        enabled: list[str],
        total: Optional[int] = None,
        rlock: Optional[threading.RLock] = None,
    ) -> None:
        """Initializes the progress tracker.

        Args:
            enabled:
                List of enabled trackers from the following list:
                ["detector_preprocess", "detector_predict", "classifier_preprocess",
                "classifier_predict", "geolocation"].
            total:
                Number of expected iterations. Optional.
            rlock:
                RLock object to use as the global tracking lock. Optional.
        """

        tqdm.monitor_interval = 0
        if rlock:
            tqdm.set_lock(rlock)

        self._pbars = {}
        if "detector_preprocess" in enabled:
            self._pbars["detector_preprocess"] = tqdm(
                desc="Detector preprocess   ",
                total=total,
                mininterval=0,
                colour="#029e73",
            )
        if "detector_predict" in enabled:
            self._pbars["detector_predict"] = tqdm(
                desc="Detector predict      ",
                total=total,
                mininterval=0,
                colour="#ece133",
            )
        if "classifier_preprocess" in enabled:
            self._pbars["classifier_preprocess"] = tqdm(
                desc="Classifier preprocess ",
                total=total,
                mininterval=0,
                colour="#d55e00",
            )
        if "classifier_predict" in enabled:
            self._pbars["classifier_predict"] = tqdm(
                desc="Classifier predict    ",
                total=total,
                mininterval=0,
                colour="#0184cb",
            )
        if "geolocation" in enabled:
            self._pbars["geolocation"] = tqdm(
                desc="Geolocation           ",
                total=total,
                mininterval=0,
                colour="#01b2b2",
            )

    def update(self, name: str) -> None:
        """Updates individual tracker.

        Args:
            name:
                Name of the individual tracker to update.
        """

        if name in self._pbars:
            self._pbars[name].update()

    def stop(self) -> None:
        """Stops progress tracking."""

        for pbar in self._pbars.values():
            pbar.close()


def _prepare_detector_input(
    detector: SpeciesNetDetector,
    filepath: str,  # input
    detector_queue: queue.Queue[DetectorInput],  # output
    exif_results: Optional[dict[str, PIL.Image.Exif]] = None,  # output
) -> None:
    """Prepares the input for detector inference.

    Responsible for loading and preprocessing an image in preparation for
    the detector. It loads the image from the provided filepath, extracts
    EXIF metadata (if requested) and then passes the image to the detector
    for preprocessing.

    Args:
        detector:
            SpeciesNetDetector to use.
        filepath:
            Path to image to load and preprocess.
        detector_queue:
            Output queue for preprocessed images for detector inference.
        exif_results:
            Output dict for extracted EXIF information. Optional.
    """

    img = load_rgb_image(filepath)
    if img is not None and exif_results is not None:
        exif_results[filepath] = img.getexif()
    try:
        img = detector.preprocess(img)
        detector_queue.put((filepath, img))
    except:
        detector_queue.put((filepath, None))
        raise


def _run_detector(
    detector: SpeciesNetDetector,
    input_queue: queue.Queue[DetectorInput],  # input
    results_dict: dict,  # output
    bboxes_queue: Optional[queue.Queue[BBoxOutput]] = None,  # output
) -> None:
    """Runs detector inference.

    Args:
        detector:
            SpeciesNetDetector to run.
        input_queue:
            Input queue of preprocessed images.
        results_dict:
            Output dict for inference results.
        bboxes_queue:
            Output queue for bounding boxes identified during inference. Optional.
    """

    filepath, img = input_queue.get()
    prediction = detector.predict(filepath, img)
    results_dict[filepath] = prediction
    if bboxes_queue:
        detections = prediction.get("detections", None)
        if detections:
            bboxes_queue.put((filepath, [BBox(*det["bbox"]) for det in detections]))
        else:
            bboxes_queue.put((filepath, []))


def _prepare_classifier_input(
    classifier: SpeciesNetClassifier,
    bboxes_queue: queue.Queue[BBoxOutput],  # input
    classifier_queue: queue.Queue[ClassifierInput],  # output
) -> None:
    """Prepares the input for classifier inference.

    Takes bounding box information from `bboxes_queue` and uses it to load and
    preprocess the image for classifier inference. It first loads the image,
    preprocesses it by potentially cropping based on bboxes and finally outputs 
    the image into `classifier_queue` to be used by the classifier model.

    Args:
        classifier:
            SpeciesNetClassifier to use.
        bboxes_queue:
            Input queue of bounding boxes from detector inference.
        classifier_queue:
            Output queue for preprocessed images for classifier inference.
    """

    filepath, bboxes = bboxes_queue.get()
    img = load_rgb_image(filepath)
    try:
        img = classifier.preprocess(img, bboxes=bboxes)
        classifier_queue.put((filepath, img))
    except:
        classifier_queue.put((filepath, None))
        raise


def _run_classifier(
    classifier: SpeciesNetClassifier,
    input_queue: queue.Queue[ClassifierInput],  # input
    results_dict: dict,  # output
) -> None:
    """Runs classifier inference.

    Takes a preprocessed image from the input queue and runs the classifier model
    on it. The output of the classifier is stored in `results_dict`.

    Args:
        classifier:
            SpeciesNetClassifier to run.
        input_queue:
            Input queue of preprocessed images.
        results_dict:
            Output dict for inference results.
    """

    filepath, img = input_queue.get()
    prediction = classifier.predict(filepath, img)
    results_dict[filepath] = prediction


def _find_admin1_region(
    filepath: str,  # input
    country: Optional[str],  # input
    latitude: Optional[float],  # input
    longitude: Optional[float],  # input
    results_dict: dict,  # output
) -> None:
    """Find the first-level administrative division for a given (lat, lon) location.

    This function uses the provided geographic information to find the first-level
    administrative division (e.g., state in the USA) using the `find_admin1_region()`
    function. The result is stored in the `results_dict`.

    Args:
        filepath:
            Image filepath.
        country:
            Country in ISO 3166-1 alpha-3 format. Optional.
        latitude:
            Float value representing latitude. Optional.
        longitude:
            Float value representing longitude. Optional.
        results_dict:
            Output dict for geolocation results.
    """

    admin1_region = find_admin1_region(country, latitude, longitude)
    results_dict[filepath] = {
        "country": country,
        "admin1_region": admin1_region,
        "latitude": latitude,
        "longitude": longitude,
    }


def _combine_results(  # pylint: disable=too-many-positional-arguments
    ensemble: SpeciesNetEnsemble,
    filepaths: list[str],
    classifier_results: dict[str, Any],
    detector_results: dict[str, Any],
    geolocation_results: dict[str, Any],
    exif_results: dict[str, PIL.Image.Exif],
    partial_predictions: dict[str, dict],
    predictions_json: Optional[StrPath] = None,
    save_lock: Optional[threading.Lock] = None,
) -> Optional[dict]:
    """Combines inference results from multiple jobs that ran independently.

    Brings together results from the classifier, detector, and geolocation steps
    to create the final predictions. The SpeciesNet ensemble model is used to
    combine these results, which may be saved to a JSON file.

    Args:
        ensemble:
            SpeciesNetEnsemble to run.
        filepaths:
            List of filepaths to ensemble predictions for.
        classifier_results:
            Dict of classifier results, with keys given by the filepaths to ensemble
            predictions for.
        detector_results:
            Dict of detector results, with keys given by the filepaths to ensemble
            predictions for.
        geolocation_results:
            Dict of geolocation results, with keys given by the filepaths to ensemble
            predictions for.
        exif_results:
            Dict of EXIF results, with keys given by the filepaths to ensemble
            predictions for.
        partial_predictions:
            Dict of partial predictions from previous ensemblings, with keys given by
            the filepaths for which predictions where already ensembled. Used to skip
            re-ensembling for the matching filepaths.
        predictions_json:
            Output filepath where to save the predictions dict in JSON format. If
            `None`, predictions are not saved to a file and are returned instead.
        save_lock:
            Threading lock used to avoid race conditions when saving predictions to a
            file. Only needed when `predictions_json` is not `None`, otherwise it is
            ignored.

    Returns:
        The predictions dict of ensembled inference results if `predictions_json` is set
        to `None`, otherwise return `None` since predictions are saved to a file.
    """

    ensemble_results = ensemble.combine(
        filepaths=filepaths,
        classifier_results=classifier_results,
        detector_results=detector_results,
        geolocation_results=geolocation_results,
        exif_results=exif_results,
        partial_predictions=partial_predictions,
    )
    predictions_dict = {"predictions": ensemble_results}
    if predictions_json:
        if save_lock:
            with save_lock:
                save_predictions(predictions_dict, predictions_json)
        else:
            save_predictions(predictions_dict, predictions_json)
    else:
        return predictions_dict


def _start_periodic_results_saving(  # pylint: disable=too-many-positional-arguments
    ensemble: SpeciesNetEnsemble,
    filepaths: list[str],
    classifier_results: dict[str, Any],
    detector_results: dict[str, Any],
    geolocation_results: dict[str, Any],
    exif_results: dict[str, PIL.Image.Exif],
    partial_predictions: dict[str, dict],
    predictions_json: StrPath,
) -> tuple[RepeatedAction, threading.Lock]:
    """Starts periodic results saving every 10 minutes.

    Initiates a background thread to save partial results periodically to avoid losing
    progress during longer inference jobs.

    Args:
        ensemble:
            SpeciesNetEnsemble to run.
        filepaths:
            List of filepaths to ensemble predictions for.
        classifier_results:
            Dict of classifier results, with keys given by the filepaths to ensemble
            predictions for.
        detector_results:
            Dict of detector results, with keys given by the filepaths to ensemble
            predictions for.
        geolocation_results:
            Dict of geolocation results, with keys given by the filepaths to ensemble
            predictions for.
        exif_results:
            Dict of EXIF results, with keys given by the filepaths to ensemble
            predictions for.
        partial_predictions:
            Dict of partial predictions from previous ensemblings, with keys given by
            the filepaths for which predictions where already ensembled. Used to skip
            re-ensembling for the matching filepaths.
        predictions_json:
            Output filepath where to save the predictions dict in JSON format.

    Returns:
        A tuple made of: (a) a repeated action to save results periodically, and (b) a
        threading lock used to avoid race conditions when saving predictions to a file.
    """

    save_lock = threading.Lock()
    periodic_saver = RepeatedAction(
        interval=600,  # 10 minutes.
        fn=_combine_results,
        ensemble=ensemble,
        filepaths=filepaths,
        classifier_results=classifier_results,
        detector_results=detector_results,
        geolocation_results=geolocation_results,
        exif_results=exif_results,
        partial_predictions=partial_predictions,
        predictions_json=predictions_json,
        save_lock=save_lock,
    )
    periodic_saver.start()
    return periodic_saver, save_lock


def _stop_periodic_results_saving(periodic_saver: Optional[RepeatedAction]) -> None:
    """Stops periodic results saving.

    Args:
        periodic_saver:
            Repeated action to stop.
    """

    if periodic_saver:
        periodic_saver.stop()
        periodic_saver.join()


def _error_callback(e: Exception) -> None:
    """Error callback to log inference errors.

    Args:
        e: Exception to log.
    """

    logging.error(
        "Unexpected inference error:\n%s",
        "".join(traceback.format_exception(type(e), e, e.__traceback__)),
    )


class SpeciesNet:
    """Main interface for running inference with SpeciesNet.

    Offers a high-level interface to run inference with the SpeciesNet model, 
    supporting various input formats. It is designed to handle full predictions
    (with both detector and classifier), classification only, or detection only tasks.
    It also can be run on a single thread, with multiple threads, or with multiple
    processes.
    """

    def __init__(
        self,
        model_name: str,
        geofence: bool = True,
        multiprocessing: bool = False,
    ) -> None:
        """Initializes the SpeciesNet model with specified settings.

        Args:
            model_name:
                 String value identifying the model to be loaded.
            geofence:
                Whether to enable geofencing or not. Defaults to `True`.
            multiprocessing:
                Whether to enable multiprocessing or not. Defaults to `False`.
        """
        if multiprocessing:
            self.manager = SyncManager()
            self.manager.start()  # pylint: disable=consider-using-with
            self.classifier = self.manager.Classifier(model_name)  # type: ignore
            self.detector = self.manager.Detector(model_name)  # type: ignore
            self.ensemble = self.manager.Ensemble(  # type: ignore
                model_name, geofence=geofence
            )
        else:
            self.manager = None
            self.classifier = SpeciesNetClassifier(model_name)
            self.detector = SpeciesNetDetector(model_name)
            self.ensemble = SpeciesNetEnsemble(model_name, geofence=geofence)

    def __del__(self) -> None:
        """Cleanup method."""
        if self.manager and hasattr(self.manager, "shutdown"):
            self.manager.shutdown()

    def _predict_using_single_thread(
        self,
        instances_dict: dict,
        progress_bars: bool = False,
        predictions_json: Optional[StrPath] = None,
    ) -> Optional[dict]:
        """Runs prediction using a single thread, processing each image one by one.

        This approach is useful for debugging or in environments where parallelization
        is not suitable. All the inference components run sequentially within the same
        thread.

        Args:
            instances_dict:
                 Instances dict to process.
            progress_bars:
                Whether to show progress bars.
            predictions_json:
                Path where to save the JSON output.

        Returns:
             The predictions dict of ensembled inference results if `predictions_json` is
            set to `None`, otherwise return `None` since predictions are saved to a file.
        """
        instances = instances_dict["instances"]
        filepaths = [instance["filepath"] for instance in instances]
        classifier_results = {}
        detector_results = {}
        geolocation_results = {}
        exif_results = {}

        # Load previously computed predictions and identify remaining instances to
        # process.
        partial_predictions, instances_to_process = load_partial_predictions(
            predictions_json, instances
        )
        num_instances_to_process = len(instances_to_process)

        # Start a periodic saver if an output file was specified.
        if predictions_json:
            periodic_saver, save_lock = _start_periodic_results_saving(
                ensemble=self.ensemble,
                filepaths=filepaths,
                classifier_results=classifier_results,
                detector_results=detector_results,
                geolocation_results=geolocation_results,
                exif_results=exif_results,
                partial_predictions=partial_predictions,
                predictions_json=predictions_json,
            )
        else:
            periodic_saver = None
            save_lock = None

        # Set up progress tracking.
        progress = Progress(
            enabled=(
                [
                    "detector_preprocess",
                    "detector_predict",
                    "classifier_preprocess",
                    "classifier_predict",
                    "geolocation",
                ]
                if progress_bars
                else []
            ),
            total=num_instances_to_process,
            rlock=threading.RLock(),
        )

        # Process instances one by one.
        for instance in instances_to_process:
            filepath = instance["filepath"]
            country = instance.get("country")
            latitude = instance.get("latitude")
            longitude = instance.get("longitude")

            # Load image.
            img = load_rgb_image(filepath)
            if img is not None:
                exif_results[filepath] = img.getexif()

            # Preprocess image for detector.
            detector_input = self.detector.preprocess(img)
            progress.update("detector_preprocess")

            # Run detector.
            detector_results[filepath] = self.detector.predict(filepath, detector_input)
            progress.update("detector_predict")

            # Preprocess image for classifier.
            detections = detector_results[filepath].get("detections", None)
            if detections:
                bboxes = [BBox(*det["bbox"]) for det in detections]
            else:
                bboxes = []
            classifier_input = self.classifier.preprocess(img, bboxes=bboxes)
            progress.update("classifier_preprocess")

            # Run classifier.
            classifier_results[filepath] = self.classifier.predict(
                filepath, classifier_input
            )
            progress.update("classifier_predict")

            # Run geolocation.
            admin1_region = find_admin1_region(country, latitude, longitude)
            geolocation_results[filepath] = {
                "country": country,
                "admin1_region": admin1_region,
                "latitude": latitude,
                "longitude": longitude,
            }
            progress.update("geolocation")

        # Stop progress tracking.
        progress.stop()

        # Stop the periodic saver if an output file was specified.
        if predictions_json:
            _stop_periodic_results_saving(periodic_saver)

        # Ensemble predictions.
        return _combine_results(
            ensemble=self.ensemble,
            filepaths=filepaths,
            classifier_results=classifier_results,
            detector_results=detector_results,
            geolocation_results=geolocation_results,
            exif_results=exif_results,
            partial_predictions=partial_predictions,
            predictions_json=predictions_json,
            save_lock=save_lock,
        )

    def _predict_using_worker_pools(  # pylint: disable=too-many-positional-arguments
        self,
        instances_dict: dict,
        progress_bars: bool = False,
        predictions_json: Optional[StrPath] = None,
        new_pool_fn: Optional[Callable] = None,
        new_list_fn: Optional[Callable] = None,
        new_dict_fn: Optional[Callable] = None,
        new_queue_fn: Optional[Callable] = None,
        new_rlock_fn: Optional[Callable] = None,
    ) -> Optional[dict]:
        """Runs prediction using worker pools (multi-threading or multiprocessing).

        This method uses worker pools to process images concurrently. This is an
        abstract method that accepts worker pools of different types.

        Args:
            instances_dict:
                Instances dict to process.
            progress_bars:
                Whether to show progress bars.
            predictions_json:
                Path where to save the JSON output.
            new_pool_fn:
                Callable that returns a new pool.
            new_list_fn:
                Callable that returns a list.
            new_dict_fn:
                Callable that returns a dict.
            new_queue_fn:
                Callable that returns a queue.
             new_rlock_fn:
                Callable that returns a thread/process lock.

        Returns:
            The predictions dict of ensembled inference results if `predictions_json` is
            set to `None`, otherwise return `None` since predictions are saved to a file.
        """
        assert new_pool_fn is not None
        assert new_list_fn is not None
        assert new_dict_fn is not None
        assert new_queue_fn is not None
        assert new_rlock_fn is not None

        instances = instances_dict["instances"]
        filepaths = new_list_fn([instance["filepath"] for instance in instances])
        classifier_results = new_dict_fn()
        detector_results = new_dict_fn()
        geolocation_results = new_dict_fn()
        exif_results = new_dict_fn()

        # Load previously computed predictions and identify remaining instances to
        # process.
        partial_predictions, instances_to_process = load_partial_predictions(
            predictions_json, instances
        )
        partial_predictions = new_dict_fn(partial_predictions)
        num_instances_to_process = len(instances_to_process)

        # Start a periodic saver if an output file was specified.
        if predictions_json:
            periodic_saver, save_lock = _start_periodic_results_saving(
                ensemble=self.ensemble,
                filepaths=filepaths,
                classifier_results=classifier_results,
                detector_results=detector_results,
                geolocation_results=geolocation_results,
                exif_results=exif_results,
                partial_predictions=partial_predictions,
                predictions_json=predictions_json,
            )
        else:
            periodic_saver = None
            save_lock = None

        # Set up progress tracking.
        progress = Progress(
            enabled=(
                [
                    "detector_preprocess",
                    "detector_predict",
                    "classifier_preprocess",
                    "classifier_predict",
                    "geolocation",
                ]
                if progress_bars
                else []
            ),
            total=num_instances_to_process,
            rlock=new_rlock_fn(),
        )

        # Set up multiprocessing pools and queues.
        common_pool = (
            new_pool_fn()
        )  # Limited by the number of logical CPUs on the machine.
        detector_pool = new_pool_fn(1)  # One single worker to run detector inference.
        classifier_pool = new_pool_fn(
            1
        )  # One single worker to run classifier inference.
        detector_queue = new_queue_fn(
            32
        )  # Limited number of images to store in memory.
        bboxes_queue = new_queue_fn()  # Unlimited number of bboxes to store in memory.
        classifier_queue = new_queue_fn(
            32
        )  # Limited number of images to store in memory.

        # Run a bunch of small tasks asynchronously.
        for instance in instances_to_process:

            # Preprocess image for detector.
            common_pool.apply_async(
                _prepare_detector_input,
                args=(
                    self.detector,
                    instance["filepath"],
                    detector_queue,
                    exif_results,
                ),
                callback=lambda _: progress.update("detector_preprocess"),
                error_callback=_error_callback,
            )

            # Preprocess image for classifier.
            common_pool.apply_async(
                _prepare_classifier_input,
                args=(self.classifier, bboxes_queue, classifier_queue),
                callback=lambda _: progress.update("classifier_preprocess"),
                error_callback=_error_callback,
            )

            # Run geolocation.
            common_pool.apply_async(
                _find_admin1_region,
                args=(
                    instance["filepath"],
                    instance.get("country"),
                    instance.get("latitude"),
                    instance.get("longitude"),
                    geolocation_results,
                ),
                callback=lambda _: progress.update("geolocation"),
                error_callback=_error_callback,
            )

        # Run inference tasks asynchronously.
        for _ in range(num_instances_to_process):

            # Run detector.
            detector_pool.apply_async(
                _run_detector,
                args=(self.detector, detector_queue, detector_results, bboxes_queue),
                callback=lambda _: progress.update("detector_predict"),
                error_callback=_error_callback,
            )

            # Run classifier.
            classifier_pool.apply_async(
                _run_classifier,
                args=(self.classifier, classifier_queue, classifier_results),
                callback=lambda _: progress.update("classifier_predict"),
                error_callback=_error_callback,
            )

        # Wait for all workers to finish.
        common_pool.close()
        detector_pool.close()
        classifier_pool.close()
        common_pool.join()
        detector_pool.join()
        classifier_pool.join()

        # Stop progress tracking.
        progress.stop()

        # Stop the periodic saver if an output file was specified.
        if predictions_json:
            _stop_periodic_results_saving(periodic_saver)

        # Ensemble predictions.
        return _combine_results(
            ensemble=self.ensemble,
            filepaths=filepaths,
            classifier_results=classifier_results,
            detector_results=detector_results,
            geolocation_results=geolocation_results,
            exif_results=exif_results,
            partial_predictions=partial_predictions,
            predictions_json=predictions_json,
            save_lock=save_lock,
        )

    def _predict_using_thread_pools(
        self,
        instances_dict: dict,
        progress_bars: bool = False,
        predictions_json: Optional[StrPath] = None,
    ) -> Optional[dict]:
        return self._predict_using_worker_pools(
            instances_dict,
            progress_bars=progress_bars,
            predictions_json=predictions_json,
            new_pool_fn=ThreadPool,
            new_list_fn=list,
            new_dict_fn=dict,
            new_queue_fn=queue.Queue,
            new_rlock_fn=threading.RLock,
        )

    def _predict_using_process_pools(
        self,
        instances_dict: dict,
        progress_bars: bool = False,
        predictions_json: Optional[StrPath] = None,
    ) -> Optional[dict]:
        assert self.manager is not None
        return self._predict_using_worker_pools(
            instances_dict,
            progress_bars=progress_bars,
            predictions_json=predictions_json,
            new_pool_fn=mp.Pool,
            new_list_fn=self.manager.list,
            new_dict_fn=self.manager.dict,
            new_queue_fn=self.manager.Queue,
            new_rlock_fn=self.manager.RLock,
        )

    def _classify_using_worker_pools(  # pylint: disable=too-many-positional-arguments
        self,
        instances_dict: dict,
        progress_bars: bool = False,
        new_pool_fn: Optional[Callable] = None,
        new_dict_fn: Optional[Callable] = None,
        new_queue_fn: Optional[Callable] = None,
        new_rlock_fn: Optional[Callable] = None,
    ) -> Optional[dict]:
        assert new_pool_fn is not None
        assert new_dict_fn is not None
        assert new_queue_fn is not None
        assert new_rlock_fn is not None

        instances = instances_dict["instances"]
        num_instances = len(instances)
        classifier_results = new_dict_fn()

        # Set up progress tracking.
        progress = Progress(
            enabled=(
                ["classifier_preprocess", "classifier_predict"] if progress_bars else []
            ),
            total=num_instances,
            rlock=new_rlock_fn(),
        )

        # Set up multiprocessing pools and queues.
        common_pool = (
            new_pool_fn()
        )  # Limited by the number of logical CPUs on the machine.
        classifier_pool = new_pool_fn(
            1
        )  # One single worker to run classifier inference.
        bboxes_queue = new_queue_fn()  # Unlimited number of bboxes to store in memory.
        classifier_queue = new_queue_fn(
            32
        )  # Limited number of images to store in memory.

        # Preprocess images for classifier.
        for instance in instances:
            bboxes_queue.put((instance["filepath"], []))
            common_pool.apply_async(
                _prepare_classifier_input,
                args=(self.classifier, bboxes_queue, classifier_queue),
                callback=lambda _: progress.update("classifier_preprocess"),
                error_callback=_error_callback,
            )

        # Run classifier.
        for _ in range(num_instances):

            classifier_pool.apply_async(
                _run_classifier,
                args=(self.classifier, classifier_queue, classifier_results),
                callback=lambda _: progress.update("classifier_predict"),
                error_callback=_error_callback,
            )

        # Wait for all workers to finish.
        common_pool.close()
        classifier_pool.close()
        common_pool.join()
        classifier_pool.join()

        # Stop progress tracking.
        progress.stop()

        return {
            "predictions": [
                classifier_results[instance["filepath"]] for instance in instances
            ]
        }

    def _classify_using_thread_pools(
        self,
        instances_dict: dict,
        progress_bars: bool = False,
    ) -> Optional[dict]:
        return self._classify_using_worker_pools(
            instances_dict,
            progress_bars=progress_bars,
            new_pool_fn=ThreadPool,
            new_dict_fn=dict,
            new_queue_fn=queue.Queue,
            new_rlock_fn=threading.RLock,
        )

    def _classify_using_process_pools(
        self,
        instances_dict: dict,
        progress_bars: bool = False,
    ) -> Optional[dict]:
        assert self.manager is not None
        return self._classify_using_worker_pools(
            instances_dict,
            progress_bars=progress_bars,
            new_pool_fn=mp.Pool,
            new_dict_fn=self.manager.dict,
            new_queue_fn=self.manager.Queue,
            new_rlock_fn=self.manager.RLock,
        )

    def _detect_using_worker_pools(  # pylint: disable=too-many-positional-arguments
        self,
        instances_dict: dict,
        progress_bars: bool = False,
        new_pool_fn: Optional[Callable] = None,
        new_dict_fn: Optional[Callable] = None,
        new_queue_fn: Optional[Callable] = None,
        new_rlock_fn: Optional[Callable] = None,
    ) -> Optional[dict]:
        assert new_pool_fn is not None
        assert new_dict_fn is not None
        assert new_queue_fn is not None
        assert new_rlock_fn is not None

        instances = instances_dict["instances"]
        num_instances = len(instances)
        detector_results = new_dict_fn()

        # Set up progress tracking.
        progress = Progress(
            enabled=(
                ["detector_preprocess", "detector_predict"] if progress_bars else []
            ),
            total=num_instances,
            rlock=new_rlock_fn(),
        )

        # Set up multiprocessing pools and queues.
        common_pool = (
            new_pool_fn()
        )  # Limited by the number of logical CPUs on the machine.
        detector_pool = new_pool_fn(1)  # One single worker to run detector inference.
        detector_queue = new_queue_fn(
            32
        )  # Limited number of images to store in memory.

        # Preprocess images for detector.
        for instance in instances:
            common_pool.apply_async(
                _prepare_detector_input,
                args=(self.detector, instance["filepath"], detector_queue),
                callback=lambda _: progress.update("detector_preprocess"),
                error_callback=_error_callback,
            )

        # Run detector.
        for _ in range(num_instances):
            detector_pool.apply_async(
                _run_detector,
                args=(self.detector, detector_queue, detector_results),
                callback=lambda _: progress.update("detector_predict"),
                error_callback=_error_callback,
            )

        # Wait for all workers to finish.
        common_pool.close()
        detector_pool.close()
        common_pool.join()
        detector_pool.join()

        # Stop progress tracking.
        progress.stop()

        return {
            "predictions": [
                detector_results[instance["filepath"]] for instance in instances
            ]
        }

    def _detect_using_thread_pools(
        self,
        instances_dict: dict,
        progress_bars: bool = False,
    ) -> Optional[dict]:
        return self._detect_using_worker_pools(
            instances_dict,
            progress_bars=progress_bars,
            new_pool_fn=ThreadPool,
            new_dict_fn=dict,
            new_queue_fn=queue.Queue,
            new_rlock_fn=threading.RLock,
        )

    def _detect_using_process_pools(
        self,
        instances_dict: dict,
        progress_bars: bool = False,
    ) -> Optional[dict]:
        assert self.manager is not None
        return self._detect_using_worker_pools(
            instances_dict,
            progress_bars=progress_bars,
            new_pool_fn=mp.Pool,
            new_dict_fn=self.manager.dict,
            new_queue_fn=self.manager.Queue,
            new_rlock_fn=self.manager.RLock,
        )

    def predict(  # pylint: disable=too-many-positional-arguments
        self,
        instances_dict: Optional[dict] = None,
        instances_json: Optional[StrPath] = None,
        filepaths: Optional[list[StrPath]] = None,
        filepaths_txt: Optional[StrPath] = None,
        folders: Optional[list[StrPath]] = None,
        folders_txt: Optional[StrPath] = None,
        run_mode: Literal[
            "single_thread", "multi_thread", "multi_process"
        ] = "multi_thread",
        progress_bars: bool = False,
        predictions_json: Optional[StrPath] = None,
    ) -> Optional[dict]:
        instances_dict = prepare_instances_dict(
            instances_dict,
            instances_json,
            filepaths,
            filepaths_txt,
            folders,
            folders_txt,
        )
        if run_mode == "single_thread":
            return self._predict_using_single_thread(
                instances_dict,
                progress_bars=progress_bars,
                predictions_json=predictions_json,
            )
        elif run_mode == "multi_thread":
            return self._predict_using_thread_pools(
                instances_dict,
                progress_bars=progress_bars,
                predictions_json=predictions_json,
            )
        elif run_mode == "multi_process":
            return self._predict_using_process_pools(
                instances_dict,
                progress_bars=progress_bars,
                predictions_json=predictions_json,
            )
        else:
            raise ValueError(f"Unknown run mode: `{run_mode}`")

    def classify(  # pylint: disable=too-many-positional-arguments
        self,
        instances_dict: Optional[dict] = None,
        instances_json: Optional[StrPath] = None,
        filepaths: Optional[list[StrPath]] = None,
        filepaths_txt: Optional[StrPath] = None,
        folders: Optional[list[StrPath]] = None,
        folders_txt: Optional[StrPath] = None,
        run_mode: Literal["multi_thread", "multi_process"] = "multi_thread",
        progress_bars: bool = False,
    ) -> Optional[dict]:
        instances_dict = prepare_instances_dict(
            instances_dict,
            instances_json,
            filepaths,
            filepaths_txt,
            folders,
            folders_txt,
        )
        if run_mode == "multi_thread":
            return self._classify_using_thread_pools(
                instances_dict,
                progress_bars=progress_bars,
            )
        elif run_mode == "multi_process":
            return self._classify_using_process_pools(
                instances_dict,
                progress_bars=progress_bars,
            )
        else:
            raise ValueError(f"Unknown run mode: `{run_mode}`")

    def detect(  # pylint: disable=too-many-positional-arguments
        self,
        instances_dict: Optional[dict] = None,
        instances_json: Optional[StrPath] = None,
        filepaths: Optional[list[StrPath]] = None,
        filepaths_txt: Optional[StrPath] = None,
        folders: Optional[list[StrPath]] = None,
        folders_txt: Optional[StrPath] = None,
        run_mode: Literal["multi_thread", "multi_process"] = "multi_thread",
        progress_bars: bool = False,
    ) -> Optional[dict]:
        instances_dict = prepare_instances_dict(
            instances_dict,
            instances_json,
            filepaths,
            filepaths_txt,
            folders,
            folders_txt,
        )
        if run_mode == "multi_thread":
            return self._detect_using_thread_pools(
                instances_dict,
                progress_bars=progress_bars,
            )
        elif run_mode == "multi_process":
            return self._detect_using_process_pools(
                instances_dict,
                progress_bars=progress_bars,
            )
        else:
            raise ValueError(f"Unknown run mode: `{run_mode}`")
