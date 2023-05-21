from __future__ import annotations

from conftest import base

from models.base import Base
    
def test_verify_primary_key(base: Base) -> None:
	"""
	It tests that the list passed as primary key is in fact a primary key. Then, by replacing the values of the primary key, the function should evaluate to `False`. 
	"""
	assert base.verify_primary_key()
	_base = base.copy()
	_base.data[_base.primary_key] = 1
	assert not _base.verify_primary_key()

