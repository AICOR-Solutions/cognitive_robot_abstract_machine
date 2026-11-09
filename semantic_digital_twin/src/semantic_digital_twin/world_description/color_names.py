"""
The names people give colors, and which colors each name covers.
"""

from __future__ import annotations

# %% Imports
from enum import StrEnum, auto

from typing_extensions import Self

from krrood.ormatic.utils import classproperty
from semantic_digital_twin.world_description.geometry import Color

# %% color names


class ColorName(StrEnum):
    """
    A color as people name it, such as the red of a red cube.

    A chromatic name covers the hues nearer its own hue than any other's; a color too
    pale or too dark to show a hue is white, grey or black by its value.
    """

    RED = auto()
    YELLOW = auto()
    GREEN = auto()
    CYAN = auto()
    BLUE = auto()
    MAGENTA = auto()
    WHITE = auto()
    BLACK = auto()
    GREY = auto()

    @classproperty
    def minimum_chromatic_saturation(cls) -> float:
        """
        The least saturation a color shows a hue with.
        """
        return 0.25

    @classproperty
    def maximum_black_value(cls) -> float:
        """
        The greatest value of a color that is black, whatever its hue.
        """
        return 0.25

    @classproperty
    def minimum_white_value(cls) -> float:
        """
        The value a color without a hue has to exceed to be white rather than grey.
        """
        return 0.5

    @property
    def representative_color(self) -> Color:
        """
        A fresh, opaque color this name covers, to give a body of this color.
        """
        match self:
            case ColorName.RED:
                return Color.RED()
            case ColorName.YELLOW:
                return Color.YELLOW()
            case ColorName.GREEN:
                return Color.GREEN()
            case ColorName.CYAN:
                return Color.CYAN()
            case ColorName.BLUE:
                return Color.BLUE()
            case ColorName.MAGENTA:
                return Color.MAGENTA()
            case ColorName.WHITE:
                return Color.WHITE()
            case ColorName.BLACK:
                return Color.BLACK()
            case ColorName.GREY:
                return Color.GREY()

    @property
    def shows_hue(self) -> bool:
        """
        Whether this name stands for a hue rather than for white, grey or black.
        """
        return self.representative_color.saturation > 0

    @classmethod
    def from_color(cls, color: Color) -> Self:
        """
        :param color: Any color.
        :return: The name that covers it.
        """
        if color.value <= cls.maximum_black_value:
            return cls.BLACK
        if color.saturation < cls.minimum_chromatic_saturation:
            return cls.WHITE if color.value > cls.minimum_white_value else cls.GREY
        return min(
            (name for name in cls if name.shows_hue),
            key=lambda name: cls._hue_distance(
                name.representative_color.hue, color.hue
            ),
        )

    @staticmethod
    def _hue_distance(first_hue: float, second_hue: float) -> float:
        """
        :return: How far apart two hues lie around the hue wheel, at most half a turn.
        """
        difference = abs(first_hue - second_hue) % 1.0
        return min(difference, 1.0 - difference)
