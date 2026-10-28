from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from geometry_msgs.msg import WrenchStamped

from giskardpy.motion_statechart.ros2_nodes.topic_monitor import TopicSubscriberNode
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from semantic_digital_twin.spatial_types import Vector3


@dataclass(eq=False, repr=False)
class ForceTorqueNode(TopicSubscriberNode[WrenchStamped]):
    """
    Superclass for all nodes that subscribe to a ROS topic that contains force and
    torque data.
    """

    msg_type: WrenchStamped = field(init=False, default=WrenchStamped)

    readings_in_agreement: int = field(default=1, kw_only=True)
    """
    How many readings in a row have to meet a monitor's condition for it to hold.

    A sensor whose noise alone reaches a threshold trips a monitor that believes a
    single reading, so a noisy one is given several readings to agree on.
    """

    _readings_so_far: int = field(init=False, default=0)
    """
    How many of the readings up to now, in a row, have met that condition.
    """

    _counted_reading: WrenchStamped | None = field(init=False, default=None)
    """
    The reading counted last, which is not counted again.

    Ticks outrun the sensor, so the same reading is seen on several of them and counting
    per tick would count one reading as several.
    """

    def enough_readings_agree(self, reading_meets_condition: bool) -> bool:
        """
        Whether enough readings in a row have met a monitor's condition.

        Counts the current reading unless it has been counted already, and starts over
        from the first reading that does not meet the condition.

        ..note:: Call this once per tick, since it is what advances the count.

        :param reading_meets_condition: Whether the current reading meets it.
        :return: Whether :attr:`readings_in_agreement` readings in a row have.
        """
        if self.current_msg is not self._counted_reading:
            self._counted_reading = self.current_msg
            self._readings_so_far = (
                self._readings_so_far + 1 if reading_meets_condition else 0
            )
        return self._readings_so_far >= self.readings_in_agreement

    def on_reset(self, context: MotionStatechartContext):
        super().on_reset(context)
        self._readings_so_far = 0
        self._counted_reading = None

    def force_as_np(self) -> np.ndarray:
        return np.array(
            [
                self.current_msg.wrench.force.x,
                self.current_msg.wrench.force.y,
                self.current_msg.wrench.force.z,
            ]
        )

    def force_magnitude(self) -> float:
        return float(np.linalg.norm(self.force_as_np()))

    def torque_as_np(self) -> np.ndarray:
        return np.array(
            [
                self.current_msg.wrench.torque.x,
                self.current_msg.wrench.torque.y,
                self.current_msg.wrench.torque.z,
            ]
        )

    def torque_magnitude(self) -> float:
        return float(np.linalg.norm(self.torque_as_np()))


@dataclass(eq=False, repr=False)
class ForceImpactMonitor(ForceTorqueNode):
    """
    This node checks if the force magnitude is above a threshold.
    """

    threshold: float = field(kw_only=True)

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        super().on_tick(context)
        if not self.has_msg():
            return ObservationStateValues.UNKNOWN
        if self.enough_readings_agree(self.force_magnitude() > self.threshold):
            return ObservationStateValues.TRUE
        return ObservationStateValues.FALSE


@dataclass(eq=False, repr=False)
class ForceDirectionMonitor(ForceTorqueNode):
    """
    Returns TRUE when the force projected onto direction exceeds a threshold.
    """

    direction: Vector3 = field(kw_only=True)
    """
    Direction in the sensor frame along which the force is measured.

    Normalized internally
    """

    threshold: float = field(kw_only=True)
    """
    Trigger threshold in N for the force projected onto direction.
    """

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        super().on_tick(context)
        if not self.has_msg():
            return ObservationStateValues.UNKNOWN
        force = Vector3.from_iterable(self.force_as_np())
        unit_direction = self.direction / self.direction.norm()
        force_along_direction = float(force.dot(unit_direction))
        if self.enough_readings_agree(force_along_direction > self.threshold):
            return ObservationStateValues.TRUE
        return ObservationStateValues.FALSE
