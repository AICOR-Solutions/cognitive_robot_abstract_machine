from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Dict, Tuple

from semantic_digital_twin.semantic_annotations.mixins import HasRootBody


@dataclass(eq=False)
class UsdSemanticLabels(HasRootBody):
    """
    The semantic labels a USD prim carries via ``UsdSemantics.LabelsAPI``, attached to
    the Body a USD-reading parser built for that prim.

    Unlike the domain-specific annotations in ``semantic_annotations.py`` (``Furniture``,
    ``Room``, ...), a taxonomy's labels are open-vocabulary strings authored by whoever
    modelled the USD asset, not a fixed set this codebase defines - this type only
    carries them, it does not interpret them.
    """

    labels: Tuple[Tuple[str, Tuple[str, ...]], ...] = field(kw_only=True)
    """
    Every taxonomy (label namespace, e.g. ``"class"``) the prim declares labels in,
    paired with the labels authored under it (e.g. ``("class", ("chair",
    "furniture"))``).

    A tuple of pairs rather than a ``dict``: a world's modification history is JSON-
    serialized, and a plain ``dict`` field has no serializer there, only the built-in
    JSON container types (``list``/``tuple``/``set``) do. Use :attr:`labels_by_taxonomy`
    for dict-style access.
    """

    @property
    def labels_by_taxonomy(self) -> Dict[str, Tuple[str, ...]]:
        """
        :return: :attr:`labels` as a taxonomy -> labels mapping.
        """
        return dict(self.labels)
