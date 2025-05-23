class WaymoLaneType:
    UNKNOWN = 0
    LANE_FREEWAY = 1
    LANE_SURFACE_STREET = 2
    LANE_BIKE_LANE = 3

    ENUM_TO_STR = {
        UNKNOWN: "LANE_UNKNOWN",
        LANE_FREEWAY: "LANE_FREEWAY",
        LANE_SURFACE_STREET: "LANE_SURFACE_STREET",
        LANE_BIKE_LANE: "LANE_BIKE_LANE",
    }

    @classmethod
    def from_waymo(cls, item):
        return cls.ENUM_TO_STR[item]


class WaymoRoadLineType:
    UNKNOWN = 0
    BROKEN_SINGLE_WHITE = 1
    SOLID_SINGLE_WHITE = 2
    SOLID_DOUBLE_WHITE = 3
    BROKEN_SINGLE_YELLOW = 4
    BROKEN_DOUBLE_YELLOW = 5
    SOLID_SINGLE_YELLOW = 6
    SOLID_DOUBLE_YELLOW = 7
    PASSING_DOUBLE_YELLOW = 8

    ENUM_TO_STR = {
        UNKNOWN: "UNKNOWN",
        BROKEN_SINGLE_WHITE: "ROAD_LINE_BROKEN_SINGLE_WHITE",
        SOLID_SINGLE_WHITE: "ROAD_LINE_SOLID_SINGLE_WHITE",
        SOLID_DOUBLE_WHITE: "ROAD_LINE_SOLID_DOUBLE_WHITE",
        BROKEN_SINGLE_YELLOW: "ROAD_LINE_BROKEN_SINGLE_YELLOW",
        BROKEN_DOUBLE_YELLOW: "ROAD_LINE_BROKEN_DOUBLE_YELLOW",
        SOLID_SINGLE_YELLOW: "ROAD_LINE_SOLID_SINGLE_YELLOW",
        SOLID_DOUBLE_YELLOW: "ROAD_LINE_SOLID_DOUBLE_YELLOW",
        PASSING_DOUBLE_YELLOW: "ROAD_LINE_PASSING_DOUBLE_YELLOW",
    }

    @classmethod
    def from_waymo(cls, item):
        return cls.ENUM_TO_STR[item]


class WaymoRoadEdgeType:
    UNKNOWN = 0
    BOUNDARY = 1
    MEDIAN = 2

    ENUM_TO_STR = {UNKNOWN: "UNKNOWN", BOUNDARY: "ROAD_EDGE_BOUNDARY", MEDIAN: "ROAD_EDGE_MEDIAN"}

    @classmethod
    def from_waymo(cls, item):
        return cls.ENUM_TO_STR[item]


class WaymoAgentType:
    UNSET = 0
    VEHICLE = 1
    PEDESTRIAN = 2
    CYCLIST = 3
    OTHER = 4

    ENUM_TO_STR = {
        UNSET: "UNSET",
        VEHICLE: "VEHICLE",
        PEDESTRIAN: "PEDESTRIAN",
        CYCLIST: "CYCLIST",
        OTHER: "OTHER",
    }

    @classmethod
    def from_waymo(cls, item):
        return cls.ENUM_TO_STR[item]
