# Constants
TILE_INNER_RADIUS_DEG = 0.1085       # Tile inner radius in degrees
TILE_OUTER_RADIUS_DEG = 0.5968       # Tile outer radius in degrees

PLATESCALE = 128.0  # um/arcsec
R_PATROL = 6.0      # mm, radius of circle that one fiber could reach
R_PATROL_DEG = R_PATROL * 1.e3 / PLATESCALE / 3600   # unit in degree

COLLISION_SEPARATION_ARCSEC = 15.625  # Fiber collision separation in arcsec, corresponding to 2mm on focalplane
COLLISION_SEPARATION_DEG = COLLISION_SEPARATION_ARCSEC / 3600.0
COLLISION_SEPARATION_MM = COLLISION_SEPARATION_ARCSEC * PLATESCALE / 1000.0


