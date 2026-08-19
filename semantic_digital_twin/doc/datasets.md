# Datasets

Semantic Digital Twin can load datasets from internet resources.
The results of the loaded datasets are completely function digital twins 
(World instances including Semantic Annotations, Kinematics, etc.).


## Sage
Scenes from [Sage](https://nvlabs.github.io/sage/) can be loaded with:

```python
from semantic_digital_twin.adapters.sage_10k_dataset.loader import Sage10kDatasetLoader

loader = Sage10kDatasetLoader()
scene = loader.create_scene(scene_url=Sage10kDatasetLoader.available_scenes()[0])
world = scene.create_world()
```

## Sapien / PartNet

Articulated assets from the [PartNet-Mobility](https://sapien.ucsd.edu/browse) dataset can be loaded with:

```python
from semantic_digital_twin.adapters.partnet_mobility_dataset.loader import PartNetMobilityDatasetLoader

loader = PartNetMobilityDatasetLoader()
world = loader.load(model_id=179) # model_id can be found at https://sapien.ucsd.edu/browse
```

Note that this requires the `sapien` library to be installed and the `SAPIEN_ACCESS_TOKEN` environment variable to be set.

## RoboCasa

Objects, fixtures, full kitchen scenes, and manipulation tasks from [RoboCasa](https://github.com/robocasa/robocasa) can be loaded with:

```python
from semantic_digital_twin.adapters.robocasa_dataset.loader import RoboCasaDatasetLoader
from semantic_digital_twin.adapters.robocasa_dataset.semantics import (
    RoboCasaKitchenApplianceCategory,
    RoboCasaObjectCategory,
)

loader = RoboCasaDatasetLoader()

kitchen_world = loader.load_kitchen(layout_id=..., style_id=...)  # a full kitchen scene
appliance_world = loader.load_kitchen_appliance(RoboCasaKitchenApplianceCategory.CABINET)  # a single appliance
object_world = loader.load_object(RoboCasaObjectCategory.APPLE)  # a single object
```

A RoboCasa task (for example `"TurnOnMicrowave"`) can be loaded together with the scene it is defined
over. `load_task` returns a `RoboCasaTask` binding the `World` to the task's natural-language
instruction, the bodies to be manipulated, and the pose the robot should start at. RoboCasa's own
robot is stripped from the world, since `semantic_digital_twin` owns the robot.

```python
task = loader.load_task("TurnOnMicrowave", layout_id=..., style_id=...)
task.instruction          # e.g. "Press the start button on the microwave."
task.manipulated_objects  # the bodies the task requires the robot to interact with
task.robot_base_pose      # where to spawn the semantic_digital_twin-owned robot
```

Note that this requires the `robocasa` and `robosuite` libraries to be installed (`robosuite` must be
installed from git, `pip install git+https://github.com/ARISE-Initiative/robosuite.git`), and the
fixture/object assets to be downloaded via `python -m robocasa.scripts.download_kitchen_assets`
(pointed at by `RoboCasaDatasetLoader.directory`, `~/robocasa-assets` by default).

## ArtVIP

Professionally modelled, articulated CAD furniture and appliances (including a dedicated IKEA furniture
category) from [ArtVIP](https://x-humanoid-artvip.github.io/) can be loaded with:

```python
from semantic_digital_twin.adapters.artvip_dataset.loader import ArtVipDatasetLoader
from semantic_digital_twin.adapters.artvip_dataset.schema import ArtVipCategory

loader = ArtVipDatasetLoader()
loader.available_objects(ArtVipCategory.IKEA_FURNITURE)  # every object name in a category

obj = loader.load(ArtVipCategory.IKEA_FURNITURE, "EKET_Cabinet_with_door_brown_walnut_effect_35x35x35cm")
obj.world  # one Body per rigid link
```

ArtVIP ships clean, hand-authored CAD meshes decomposed into rigid links connected by real USD Physics
joints, each with an authored axis, frame, and limit read directly from the object's USD file.
`RevoluteConnection`/`PrismaticConnection` are used for links with a joint of the matching type,
`FixedConnection` for everything else. The USD stage itself is parsed by the general-purpose
`semantic_digital_twin.adapters.usd.USDParser`, the USD counterpart to `URDFParser`/`MJCFParser`; this
loader only handles what is ArtVIP-specific - discovering, downloading, and disambiguating an object's
files on Hugging Face.

The catalog is 450 objects across the 9 categories in `ArtVipCategory`. Some categories nest an extra
subcategory level on Hugging Face (e.g. a `MAJOR_APPLIANCES` object under
`major_appliances/refrigerator/fridge/`) - `available_objects` returns each object's path relative to its
category, which may include that extra segment, not just a single name.

Loading every object in the catalog into a `World`, and building each of those into
`semantic_digital_twin.adapters.multi_sim.MujocoSim`, was verified directly (not just the two objects
originally used to build the loader): all 450 load, and all 450 build successfully. Real ArtVIP data
occasionally strays from what a single object's authored USD might suggest is a safe assumption - a
joint's `body0` can target a link's mesh prim directly instead of its enclosing `Xform`, and a
mirrored part's authored joint limits can have lower > upper - both handled rather than left to break
the built `World`. Real CAD furniture also regularly includes thin panel geometry (a door slab, a
backing panel) MuJoCo's default mesh-inertia
estimation rejects; this loader's build now goes through fixes to the shared MuJoCo pipeline
(`multi_sim.py`) that every dataset loader benefits from, not just ArtVIP's.

The dataset is public (Apache 2.0) and requires no gated access. Note that this requires the `usd-core`
library (`pxr`) to be installed to read the object's USD stage.
