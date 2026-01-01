"""
Window Utilities for Forest Representation Model

Shared window classes for defining spatial and temporal regions
to read from Zarr datasets.

These are used by both DataReader and MaskBuilder.
"""

from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class SpatialWindow:
    """
    Represents a spatial window for data reading.
    
    Defines a rectangular region in pixel coordinates with:
    - Upper-left corner at (row_start, col_start)
    - Dimensions (height, width)
    """
    row_start: int
    col_start: int
    height: int
    width: int
    
    @classmethod
    def from_upper_left_and_hw(
        cls, 
        upper_left: Tuple[int, int], 
        hw: Tuple[int, int]
    ) -> 'SpatialWindow':
        """
        Create spatial window from upper-left corner and dimensions.
        
        Args:
            upper_left: (row, col) coordinates of upper-left corner
            hw: (height, width) dimensions
            
        Returns:
            SpatialWindow object
            
        Example:
            >>> window = SpatialWindow.from_upper_left_and_hw(
            ...     upper_left=(100, 200),
            ...     hw=(256, 256)
            ... )
            >>> window.row_start
            100
        """
        return cls(
            row_start=upper_left[0],
            col_start=upper_left[1],
            height=hw[0],
            width=hw[1]
        )
    
    @classmethod
    def from_bounds(
        cls,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int
    ) -> 'SpatialWindow':
        """
        Create spatial window from explicit bounds.
        
        Args:
            row_start: Starting row (inclusive)
            row_end: Ending row (exclusive)
            col_start: Starting column (inclusive)
            col_end: Ending column (exclusive)
            
        Returns:
            SpatialWindow object
            
        Example:
            >>> window = SpatialWindow.from_bounds(0, 256, 0, 256)
            >>> window.height
            256
        """
        return cls(
            row_start=row_start,
            col_start=col_start,
            height=row_end - row_start,
            width=col_end - col_start
        )
    
    @property
    def row_end(self) -> int:
        """Ending row (exclusive)"""
        return self.row_start + self.height
    
    @property
    def col_end(self) -> int:
        """Ending column (exclusive)"""
        return self.col_start + self.width
    
    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        """Return (row_start, row_end, col_start, col_end)"""
        return (self.row_start, self.row_end, self.col_start, self.col_end)
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Return (height, width)"""
        return (self.height, self.width)
    
    @property
    def size(self) -> int:
        """Total number of pixels in window"""
        return self.height * self.width
    
    def to_slice(self) -> Tuple[slice, slice]:
        """
        Convert to (row_slice, col_slice) for array indexing.
        
        Returns:
            Tuple of slice objects for numpy/zarr indexing
            
        Example:
            >>> window = SpatialWindow(100, 200, 256, 256)
            >>> row_slice, col_slice = window.to_slice()
            >>> data = array[row_slice, col_slice]
        """
        return (
            slice(self.row_start, self.row_end),
            slice(self.col_start, self.col_end)
        )
    
    def intersects(self, other: 'SpatialWindow') -> bool:
        """
        Check if this window intersects with another.
        
        Args:
            other: Another SpatialWindow
            
        Returns:
            True if windows overlap, False otherwise
        """
        return not (
            self.row_end <= other.row_start or
            self.row_start >= other.row_end or
            self.col_end <= other.col_start or
            self.col_start >= other.col_end
        )
    
    def intersection(self, other: 'SpatialWindow') -> 'SpatialWindow':
        """
        Compute intersection of this window with another.
        
        Args:
            other: Another SpatialWindow
            
        Returns:
            SpatialWindow representing the intersection
            
        Raises:
            ValueError: If windows don't intersect
        """
        if not self.intersects(other):
            raise ValueError("Windows do not intersect")
        
        row_start = max(self.row_start, other.row_start)
        row_end = min(self.row_end, other.row_end)
        col_start = max(self.col_start, other.col_start)
        col_end = min(self.col_end, other.col_end)
        
        return SpatialWindow.from_bounds(row_start, row_end, col_start, col_end)
    
    def __repr__(self) -> str:
        return (
            f"SpatialWindow(row={self.row_start}:{self.row_end}, "
            f"col={self.col_start}:{self.col_end}, "
            f"shape={self.height}x{self.width})"
        )


@dataclass
class TemporalWindow:
    """
    Represents a temporal window for data reading.
    
    Defines a contiguous time period with:
    - end_year: The final year in the window (inclusive)
    - window_length: Number of years in the window
    
    The window extends backwards from end_year.
    
    Example:
        TemporalWindow(end_year=2020, window_length=10)
        represents years 2011-2020 (10 years)
    """
    end_year: int
    window_length: int
    
    @property
    def start_year(self) -> int:
        """First year in the window (inclusive)"""
        return self.end_year - self.window_length + 1
    
    @property
    def years(self) -> List[int]:
        """
        All years in the window as a list.
        
        Returns:
            List of years from start_year to end_year (inclusive)
            
        Example:
            >>> window = TemporalWindow(end_year=2020, window_length=3)
            >>> window.years
            [2018, 2019, 2020]
        """
        return list(range(self.start_year, self.end_year + 1))
    
    @property
    def year_range(self) -> Tuple[int, int]:
        """Return (start_year, end_year) tuple"""
        return (self.start_year, self.end_year)
    
    @classmethod
    def from_year_range(cls, start_year: int, end_year: int) -> 'TemporalWindow':
        """
        Create temporal window from explicit year range.
        
        Args:
            start_year: First year (inclusive)
            end_year: Last year (inclusive)
            
        Returns:
            TemporalWindow object
            
        Example:
            >>> window = TemporalWindow.from_year_range(2011, 2020)
            >>> window.window_length
            10
        """
        window_length = end_year - start_year + 1
        return cls(end_year=end_year, window_length=window_length)
    
    def contains_year(self, year: int) -> bool:
        """
        Check if a year is within this window.
        
        Args:
            year: Year to check
            
        Returns:
            True if year is in [start_year, end_year], False otherwise
        """
        return self.start_year <= year <= self.end_year
    
    def overlaps(self, other: 'TemporalWindow') -> bool:
        """
        Check if this window overlaps with another.
        
        Args:
            other: Another TemporalWindow
            
        Returns:
            True if windows overlap, False otherwise
        """
        return not (
            self.end_year < other.start_year or
            self.start_year > other.end_year
        )
    
    def shift(self, years: int) -> 'TemporalWindow':
        """
        Create a new window shifted by a number of years.
        
        Args:
            years: Number of years to shift (positive = forward, negative = backward)
            
        Returns:
            New TemporalWindow with shifted end_year
            
        Example:
            >>> window = TemporalWindow(end_year=2020, window_length=10)
            >>> earlier = window.shift(-5)
            >>> earlier.end_year
            2015
        """
        return TemporalWindow(
            end_year=self.end_year + years,
            window_length=self.window_length
        )
    
    def expand(self, years: int) -> 'TemporalWindow':
        """
        Create a new window with expanded length.
        
        Args:
            years: Number of years to add to window_length
            
        Returns:
            New TemporalWindow with increased window_length
            
        Example:
            >>> window = TemporalWindow(end_year=2020, window_length=10)
            >>> longer = window.expand(5)
            >>> longer.window_length
            15
            >>> longer.start_year
            2006
        """
        return TemporalWindow(
            end_year=self.end_year,
            window_length=self.window_length + years
        )
    
    def to_index_range(self, base_year: int = 1985) -> Tuple[int, int]:
        """
        Convert to 0-based index range for array indexing.
        
        Args:
            base_year: The year corresponding to index 0 (default: 1985)
            
        Returns:
            Tuple of (start_index, end_index) for array slicing
            
        Example:
            >>> window = TemporalWindow(end_year=2020, window_length=10)
            >>> window.to_index_range(base_year=1985)
            (26, 36)  # 2011 is index 26, 2020 is index 35, so slice [26:36]
        """
        start_index = self.start_year - base_year
        end_index = self.end_year - base_year + 1  # +1 for exclusive end
        return (start_index, end_index)
    
    def __repr__(self) -> str:
        return (
            f"TemporalWindow({self.start_year}-{self.end_year}, "
            f"{self.window_length}y)"
        )
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, TemporalWindow):
            return False
        return (
            self.end_year == other.end_year and
            self.window_length == other.window_length
        )
    
    def __hash__(self) -> int:
        return hash((self.end_year, self.window_length))


if __name__ == '__main__':
    # Example usage and tests
    print("=" * 70)
    print("SpatialWindow Examples")
    print("=" * 70)
    
    # Create spatial window
    spatial = SpatialWindow.from_upper_left_and_hw(
        upper_left=(1000, 2000),
        hw=(256, 256)
    )
    print(f"\n{spatial}")
    print(f"Bounds: {spatial.bounds}")
    print(f"Shape: {spatial.shape}")
    print(f"Size: {spatial.size:,} pixels")
    
    # Create from bounds
    spatial2 = SpatialWindow.from_bounds(0, 512, 0, 512)
    print(f"\n{spatial2}")
    
    # Check intersection
    print(f"\nWindows intersect? {spatial.intersects(spatial2)}")
    if spatial.intersects(spatial2):
        intersection = spatial.intersection(spatial2)
        print(f"Intersection: {intersection}")
    
    print("\n" + "=" * 70)
    print("TemporalWindow Examples")
    print("=" * 70)
    
    # Create temporal window
    temporal = TemporalWindow(end_year=2020, window_length=10)
    print(f"\n{temporal}")
    print(f"Start year: {temporal.start_year}")
    print(f"End year: {temporal.end_year}")
    print(f"Years: {temporal.years}")
    print(f"Year range: {temporal.year_range}")
    
    # Create from year range
    temporal2 = TemporalWindow.from_year_range(2015, 2024)
    print(f"\n{temporal2}")
    
    # Check overlap
    print(f"\nWindows overlap? {temporal.overlaps(temporal2)}")
    
    # Shift and expand
    shifted = temporal.shift(-5)
    print(f"\nShifted -5 years: {shifted}")
    
    expanded = temporal.expand(5)
    print(f"Expanded +5 years: {expanded}")
    
    # Index conversion
    start_idx, end_idx = temporal.to_index_range(base_year=1985)
    print(f"\nIndex range (base_year=1985): [{start_idx}:{end_idx}]")
    print(f"This would select: array[{start_idx}:{end_idx}]")
