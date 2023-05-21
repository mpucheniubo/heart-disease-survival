from __future__ import annotations

from copy import deepcopy
import logging
import pandas as pd

class Base:
	"""
	This class serves as a base to handel data structures. It has the following attributes/properties:

	data: `DataFrame`
		The data set for the instance.
	primary_key: `list[str]`
		List with columns of the data that represent a unique record.
	"""
	def __init__(self, data: pd.DataFrame | None = None, primary_key: list[str] | None = None) -> None:
		self.data: pd.DataFrame = data if data is not None else pd.DataFrame()
		self.primary_key: list[str] = primary_key or []

	def copy(self) -> Base:
		"""
		This method provides a deepcopy of the object instance.

		# Return

		It outputs a new instance of the `Base` that has been copied.
		"""
		logging.info("Creating copy of 'Base' instance.")
		return deepcopy(self)

	def verify_primary_key(self) -> bool:
		"""
		This method verifies that the primary key attribute represents in fact a unique record.

		# Return

		It outputs `True` if the primary key is verified, `False` otherwise.
		"""
		logging.info("Verifying primary key of data.")
		return self.data.drop_duplicates(subset=self.primary_key).shape[0] == self.data.shape[0]