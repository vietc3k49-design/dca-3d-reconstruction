GENERIC RECONSTRUCTION -- CURRENT IMPLEMENTATION

Primary entry: reconstruct_images.py, same algorithm for arbitrary images.
No star/can/cylinder model, known dimensions or measured camera parameters.
It uses SIFT correspondences, masks, free mesh vertices, unknown cameras,
small pixel-normalized shape priors and IPC contact feasibility.

Tested heavy environment: Python 3.12 on 3090 fallback, D:/moge/.venv.
IPC/libigl were installed ONLY under:
  G:/research_jobs/road_geometry_20261114_contact/vendor
Set PYTHONPATH to that directory for the native optional dependencies.
No torch/model-weight changes were made to the shared MoGe environment.

Raw image entry (replace paths for any object):
  python reconstruct_images.py --images photo1.webp photo2.webp --output result --point-tolerance-pixels 1 --hidden-completion-steps 900
  Supply --masks when background extraction is unsuitable.
  Default contact-mode is ipc; diagnostic mode is the INCOMPLETE old guard.
  Default field interpolation is cubic. Camera-prior coefficient is 0.001
  in pixel-squared units; --calibration-prior-pixels is its square root.
  Point tolerance 1 is an uncertainty assumption; 0 reproduces squared loss.

Saved-problem research paths:
  run_contact_fit.py --source CASE --output NEW --camera-coefficient 0.001
  run_hidden_completion.py --source CASE --output NEW --steps 900
The latter changes only model-hidden unanchored vertices. It uses the SAME
objective and retains protected feature projections, observations and cameras
up to the similarity gauge. It is NOT a new object-shape template.

Numerical tests: check_point_uncertainty.py, check_smooth_fields.py,
check_contact.py, check_contact_integration.py, check_hidden_completion.py.
Native contact tests require the tested Python 3.12 environment and vendor path.
Exact code and input manifests are saved with each completed remote job.

Scientific limits: no measured 3D truth or metric scale; cropped web photos
not verified as one rigid capture; fixed sphere topology; unknown calibration;
mask/sparse-feature fitting is not full RGB matching or a visible-surface
bijection proof. Unseen protrusions remain. Small priors do not enforce exact
symmetry/convexity, uniqueness or global optimality. This is NOT DCA and not
a trained neural reconstruction model. See the complete experiment report.
