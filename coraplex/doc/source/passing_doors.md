# Passing Doors and Riding Elevators

A robot cannot open every door itself. {class}`~coraplex.robot_plans.actions.core.navigation.PassDoorAction` instead
drives the robot up to a doorway, waits until its lidar sees the doorway open, and only then drives through.
{class}`~coraplex.robot_plans.actions.core.navigation.ElevatorNavigation` passes the elevator doors twice this way, once
to board the cabin and once to leave it.

## Passing a Doorway

{class}`~coraplex.robot_plans.actions.core.navigation.PassDoorAction` takes the
{class}`~semantic_digital_twin.semantic_annotations.semantic_annotations.Door` annotations covering a doorway, such as
both leaves of a double door, and a point on the side the robot should end up on. All poses are measured in the frame of
the first door's entry way, whose x-axis points through the doorway. The action runs three steps:

1. Drive to a pose {attr}`~coraplex.robot_plans.actions.core.navigation.PassDoorAction.approach_distance` in front of
   the doorway's centre, on the side away from the destination, facing the doorway.
2. Optionally say {attr}`~coraplex.robot_plans.actions.core.navigation.PassDoorAction.waiting_announcement`, for
   example to ask a person to open the door.
3. Drive to a pose {attr}`~coraplex.robot_plans.actions.core.navigation.PassDoorAction.exit_distance` beyond the
   doorway. This step sits in a {class}`~cramph.composites.PausedUntilTrue`, so it is held until the doorway monitor
   observes True.

The action needs a mobile base with a lidar
({class}`~semantic_digital_twin.robots.robot_part_mixins.HasLidar`); otherwise it raises
{class}`~semantic_digital_twin.robots.exceptions.MissingLidarError`.

## The Doorway Monitor

The monitor is a {class}`~giskardpy.motion_statechart.monitors.sensor_monitors.PassageClear` named `DoorwayClear`. Its
passage is a line segment across the centre of the doorway, as wide as the base's diameter (the base radius to each
side of the centre) and at the height of the base, so it lies in the lidar's scan plane.

On every tick it reads the lidar and looks at the beams that cross that segment:

| Situation                                                        | Observation |
|------------------------------------------------------------------|-------------|
| No beam crosses the segment                                      | Unknown     |
| A crossing beam hits something before or within 5 cm past it     | False       |
| Every crossing beam travels further than 5 cm past the segment   | True        |

The 5 cm are the monitor's {attr}`~giskardpy.motion_statechart.monitors.sensor_monitors.PassageClear.tolerance`. As
{class}`~cramph.composites.PausedUntilTrue` holds its node for anything but True, Unknown keeps the robot waiting, too.

Two flags change this behaviour, and the action sets them as follows:

- {attr}`~giskardpy.motion_statechart.monitors.sensor_monitors.PassageClear.stays_clear_once_seen` is always on. Once
  the robot drives into the doorway, its lidar no longer looks through the passage, and without this flag the monitor
  would pause the robot halfway through.
- {attr}`~giskardpy.motion_statechart.monitors.sensor_monitors.PassageClear.clear_only_after_blocked` follows
  {attr}`~coraplex.robot_plans.actions.core.navigation.PassDoorAction.doorway_closes_first`. With it, the monitor
  observes False even through an open doorway until it has seen the doorway blocked once since it started. This is for
  doors that are still open from before and have to close and open again first.

Both remembered states are reset whenever the monitor starts.

## Riding an Elevator

{class}`~coraplex.robot_plans.actions.core.navigation.ElevatorNavigation` strings door passing and speech together:

1. Say where the robot is and where it is headed
   ({attr}`~coraplex.robot_plans.actions.core.navigation.ElevatorRideAnnouncement.DEPARTING`).
2. **Board**: pass the elevator doors towards the cabin's centre. The robot waits
   {attr}`~coraplex.robot_plans.actions.core.navigation.ElevatorNavigation.exit_clearance` in front of the doors, asks
   for them to be opened
   ({attr}`~coraplex.robot_plans.actions.core.navigation.ElevatorRideAnnouncement.WAITING_AT_DOORS`), and drives in
   as soon as the monitor sees them open.
3. Attach the robot to the cabin, so the cabin carries it during the ride.
4. Turn in the middle of the cabin to face the doors, within
   {attr}`~coraplex.robot_plans.actions.core.navigation.ElevatorNavigation.cabin_pose_threshold`.
5. Ask for the button of the target floor to be pressed
   ({attr}`~coraplex.robot_plans.actions.core.navigation.ElevatorRideAnnouncement.RIDING`).
6. **Leave**: pass the doors towards a point a cabin depth outside, with
   {attr}`~coraplex.robot_plans.actions.core.navigation.PassDoorAction.doorway_closes_first` on. The doors are still
   open from boarding, so the monitor first waits for them to close and then to open again. Once it sees them open, the
   robot is attached back to what it hung from before the ride and drives out.
7. Announce the arrival
   ({attr}`~coraplex.robot_plans.actions.core.navigation.ElevatorRideAnnouncement.ARRIVED`).

```{warning}
The robot leaves at the first floor the doors open on after closing, whether or not it is the target floor. Nothing
checks which floor the cabin stopped at.
```
