# To Do

1. Migrate the Localizer from `yolov5 <= 6.0.7` to `ultralytics`.
   - Do it after the tests exist, and keep the `Localizer` output
     (`label, confidence, x_p, y_p, bbox, edge`) and the `bbox` convention
     used by `CATCHapp.crop` unchanged.
   - The YOLOv5 weights in `cfg_yolov5/` cannot be loaded by ultralytics, so
     the Localizer must be retrained; fold this into the training rebuild.
   - License decided: CATCH keeps its MIT license (an open-source community
     project that relies on YOLO), and the AGPL-3.0 of ultralytics is
     accepted.  Users who distribute or serve a combination of CATCH and
     ultralytics are bound by the AGPL for that combination.
   - Compare with the pylorenzmie Localizer before choosing a detector.
2. Fix yolov5 training script default options (moot after item 1).
3. Clean up imports; the dependencies are in `pyproject.toml` now.
4. Test `EstimatorTraining.py` and `train_yolov5.py`; they are omitted from
   coverage and have not been run since the port of data generation.
5. `examples/pylm_fit.py` imports modules that no longer exist
   (`CATCH.Estimator_Torch`, `LMHologram`, `coordinates`); port it to
   pylorenzmie 3.x or delete it.
