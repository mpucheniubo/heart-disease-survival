from __future__ import annotations

from models.base import Base


def test_verify_primary_key(base: Base) -> None:
    """
    It tests that the list passed as primary key is in fact a primary key. Then, by replacing the values of the primary key, the function should evaluate to `False`.
    """
    assert base.verify_primary_key()
    _base = base.copy()
    _base.data[_base.primary_key] = 1
    assert not _base.verify_primary_key()


def test_base___bool__(base: Base) -> None:
    """
    It tests that the `__bool__` method evaluates to `True` when both the primary key is verified and there is data. Then, by removing the data, it tests that it evaluates to
    `False`.
    """
    assert base
    _base = base.copy()
    _base.data = _base.data.head(0)
    assert not _base


def test_drop_columns_with_one_value(base: Base) -> None:
    """
    It tests that columns with only one value are dropped.
    """
    _base = base.copy()
    _base.drop_columns_with_one_value()
    assert set(base.data.columns) - set(_base.data.columns) == {"c"}


def test_set_booleans(base: Base) -> None:
    """
    It tests that object columns containing two entities are converted to `bool` data type and the mapping is filled accordingly.
    """
    _base = base.copy()
    _base.set_booleans()
    assert set(_base.data.select_dtypes(include="bool").columns) == {"f", "g"}
    assert _base.boolean_mapping.get("f") == {"N": False, "Y": True}
    assert _base.boolean_mapping.get("g") == {"0": False, "1": True}


def test_set_categories(base: Base) -> None:
    """
    It tests that the object columns containg three or more entities are converted to `category` data type.
    """
    _base = base.copy()
    _base.set_categories()
    assert set(_base.data.select_dtypes(include="category").columns) == {"d", "e"}
