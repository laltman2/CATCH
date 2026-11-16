# To Do

1. Migrate the Localizer from `yolov5 <= 6.0.7` to `ultralytics`.
   - Do it after the tests exist, and keep the `Localizer` output
     (`label, confidence, x_p, y_p, bbox, edge`) and the `bbox` convention
     used by `CATCHapp.crop` unchanged.
   - The YOLOv5 weights in `cfg_yolov5/` cannot be loaded by ultralytics, so
     the Localizer must be retrained; fold this into the training rebuild.
   - License decided: CATCH stays GPL-3.0-or-later (open-source community
     project that relies on YOLO), so the AGPL-3.0 of ultralytics is accepted.
     Revisit only if the dependency becomes a problem for users.
   - Compare with the pylorenzmie Localizer before choosing a detector.
2. Fix yolov5 training script default options (moot after item 1).
3. Clean up imports; the dependencies are in `pyproject.toml` now.
