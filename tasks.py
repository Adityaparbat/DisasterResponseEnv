# tasks.py
"""
Disaster Response Tasks — RL/MDP Version
Clean configuration-only file. Scenarios and Graders are now handled dynamically
by the core Environment logic in env.py.
"""

TASKS = {

    "single_incident_response": {
        "id": "single_incident_response",
        "difficulty": "easy",
        "max_steps": 3,
        "resources_manifest": {
            "fire_truck_1":  "fire_truck",
            "fire_truck_2":  "fire_truck",
            "ambulance_1":   "ambulance",
            "ambulance_2":   "ambulance",
            "police_unit_1": "police",
            "police_unit_2": "police"
        },
        "initial_resources": [
            "fire_truck_1", "fire_truck_2",
            "ambulance_1", "ambulance_2",
            "police_unit_1", "police_unit_2"
        ],
        "initial_incidents": [
            {
                "id": "INC-001",
                "type": "structural_fire",
                "severity": "critical",
                "requires": ["fire_truck", "ambulance"],
                "location": "45 Oak Street",
                "time_to_resolve": 2
            },
            {
                "id": "INC-002",
                "type": "road_hazard",
                "severity": "low",
                "requires": ["police"],
                "location": "Bridge Road",
                "time_to_resolve": 1
            }
        ],
        "constraints": [
            "Send exactly 1 fire_truck and 1 ambulance to INC-001.",
            "Send exactly 1 police unit to INC-002.",
            "Total dispatch: 3 units maximum."
        ]
    },

    "multi_incident_triage": {
        "id": "multi_incident_triage",
        "difficulty": "medium",
        "max_steps": 3,
        "resources_manifest": {
            "unit_alpha":   "ambulance",
            "unit_bravo":   "police",
            "unit_charlie": "police",
            "unit_delta":   "fire_truck",
            "unit_echo":    "ambulance"
        },
        "initial_resources": [
            "unit_alpha", "unit_bravo", "unit_charlie", "unit_delta", "unit_echo"
        ],
        "initial_incidents": [
            {
                "id": "INC-001",
                "type": "cardiac_arrest",
                "severity": "critical",
                "requires": ["ambulance"],
                "location": "City Hospital lobby",
                "time_to_resolve": 3
            },
            {
                "id": "INC-002",
                "type": "vehicle_collision",
                "severity": "moderate",
                "requires": ["police"],
                "location": "Highway 7",
                "time_to_resolve": 2
            },
            {
                "id": "INC-003",
                "type": "false_alarm",
                "severity": "none",
                "requires": [],
                "location": "88 Maple Drive",
                "time_to_resolve": 1
            }
        ],
        "constraints": [
            "unit_echo is on mandatory maintenance — DO NOT dispatch.",
            "INC-003 is a confirmed false alarm — zero units required.",
            "Check resources_manifest carefully: units have coded names."
        ],
        "unavailable_units": ["unit_echo"]
    },

    "dynamic_escalation": {
        "id": "dynamic_escalation",
        "difficulty": "hard",
        "max_steps": 4,
        "resources_manifest": {
            "fire_truck_1":         "fire_truck",
            "fire_truck_2":         "fire_truck",
            "fire_truck_3":         "fire_truck",
            "hazmat_unit_1":        "hazmat",
            "hazmat_unit_2":        "hazmat",
            "ambulance_1":          "ambulance",
            "ambulance_2":          "ambulance",
            "police_unit_1":        "police",
            "police_unit_2":        "police",
            "mental_health_unit_1": "mental_health"
        },
        "initial_resources": [
            "fire_truck_1", "fire_truck_2", "fire_truck_3",
            "hazmat_unit_1", "hazmat_unit_2",
            "ambulance_1", "ambulance_2",
            "police_unit_1", "police_unit_2",
            "mental_health_unit_1"
        ],
        "initial_incidents": [
            {
                "id": "INC-001",
                "type": "gas_leak",
                "severity": "critical",
                "requires": ["hazmat", "fire_truck"],
                "location": "Riverside Mall",
                "time_to_resolve": 4
            },
            {
                "id": "INC-002",
                "type": "psychiatric_crisis",
                "severity": "moderate",
                "requires": ["mental_health", "police"],
                "location": "8 Pine Avenue",
                "time_to_resolve": 3
            }
        ],
        "constraints": [
            "fire_truck_3 is UNFIT FOR SERVICE — DO NOT dispatch.",
            "hazmat_unit_1 is UNAVAILABLE (Route 9) — DO NOT dispatch.",
            "Ambulances are strictly forbidden from INC-001 (Gas Leak zone).",
            "Reserve 1 fire_truck for potential escalation."
        ],
        "unavailable_units": ["fire_truck_3", "hazmat_unit_1"],
        "forbidden_dispatches": {
            "INC-001": ["ambulance"]
        }
    },

    "citywide_crisis_management": {
        "id": "citywide_crisis_management",
        "difficulty": "master",
        "max_steps": 5,
        "resources_manifest": {
            "unit_m1": "marine_unit", "unit_m2": "marine_unit",
            "unit_a1": "aviation", "unit_a2": "aviation", "unit_a3": "aviation", "unit_a4": "aviation",
            "unit_med1": "ambulance", "unit_med2": "ambulance", "unit_med3": "ambulance", "unit_med4": "ambulance", "unit_med5": "ambulance",
            "unit_p1": "police", "unit_p2": "police", "unit_p3": "police", "unit_p4": "police", "unit_p5": "police", "unit_p6": "police",
            "unit_b1": "bomb_squad", "unit_b2": "bomb_squad",
            "unit_h1": "hazmat", "unit_h2": "hazmat", "unit_h3": "hazmat",
            "unit_f1": "fire_truck", "unit_f2": "fire_truck", "unit_f3": "fire_truck", "unit_delta_4": "fire_truck"
        },
        "initial_resources": [
            "unit_m1", "unit_m2",
            "unit_a1", "unit_a2", "unit_a3", "unit_a4",
            "unit_med1", "unit_med2", "unit_med3", "unit_med4", "unit_med5",
            "unit_p1", "unit_p2", "unit_p3", "unit_p4", "unit_p5", "unit_p6",
            "unit_b1", "unit_b2",
            "unit_h1", "unit_h2", "unit_h3",
            "unit_f1", "unit_f2", "unit_f3", "unit_delta_4"
        ],
        "initial_incidents": [
            {
                "id": "INC-001", "type": "bridge_collapse", "severity": "critical", 
                "requires": ["marine_unit", "aviation", "ambulance", "police"], "location": "Bridge",
                "time_to_resolve": 4
            },
            {
                "id": "INC-002", "type": "hostage_situation", "severity": "critical", 
                "requires": ["police", "bomb_squad", "ambulance"], "location": "City Bank",
                "time_to_resolve": 4
            },
            {
                "id": "INC-003", "type": "electrical_fire", "severity": "critical", 
                "requires": ["fire_truck", "hazmat"], "location": "Server Farm",
                "time_to_resolve": 3
            },
            {
                "id": "INC-004", "type": "chemical_spill", "severity": "moderate", 
                "requires": ["hazmat", "police"], "location": "Route 6",
                "time_to_resolve": 2
            }
        ],
        "constraints": [
            "Keep exactly 2 aviation units in the reserved_units list.",
            "Do not dispatch unit_med3, unit_p4, or unit_h2 (Maintenance).",
            "Only unit_delta_4 is compatible with the Server Farm chemical fire."
        ],
        "unavailable_units": ["unit_med3", "unit_p4", "unit_h2"],
        "identity_locked_units": {
            "INC-003": "unit_delta_4"
        }
    }
}