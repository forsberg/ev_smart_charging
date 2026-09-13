"""Global scheduler for serial EV charging with price optimization"""

from dataclasses import dataclass, field
from math import ceil
import asyncio
from copy import deepcopy
from datetime import datetime
import logging
from typing import Any, Optional, cast

from homeassistant.util import dt

from custom_components.ev_smart_charging import EVSmartConfigEntry
from custom_components.ev_smart_charging.const import (
    READY_QUARTER_NONE,
    START_QUARTER_NONE,
)
from custom_components.ev_smart_charging.helpers.raw import Raw
from custom_components.ev_smart_charging.helpers.general import Utils

_LOGGER = logging.getLogger(__name__)


class EVData:
    """Data structure for EV requirements"""

    def __init__(
        self,
        ev_id: str,
        name: str,
        soc_current: float,
        soc_target: float,
        soc_min: float,
        speed: float,
        start_quarter: int,
        ready_quarter: int,
        priority: int,
        max_price: float,
        apply_limit: bool,
        connected: bool,
        active: bool,
    ):
        self.ev_id = ev_id
        self.name = name
        self.soc_current = soc_current
        self.soc_target = soc_target
        self.soc_min = soc_min
        self.speed = speed
        self.start_quarter = start_quarter
        self.ready_quarter = ready_quarter
        self.priority = priority
        self.max_price = max_price
        self.apply_limit = apply_limit
        self.connected = connected
        self.active = active

        # Calculate requirements
        self.required_for_target = self._calc_quarters(soc_target)
        self.required_for_min = self._calc_quarters(soc_min)
        self.required = max(self.required_for_target, self.required_for_min)

    def _calc_quarters(self, target_soc: float) -> int:
        """Calculate quarters needed to reach target SOC"""
        if self.speed <= 0:
            return 0
        needed_pct = target_soc - self.soc_current
        if needed_pct <= 0:
            return 0
        return ceil(needed_pct / self.speed * 4)

    def get_eligible_quarters(
        self, all_quarters: list[dict], current_quarter: int
    ) -> list[int]:
        """Get list of quarter indices this EV can use"""
        eligible = []

        for i, q in enumerate(all_quarters):
            q_quarter = Utils.datetime_quarter(dt.as_local(q["start"]))

            # Must be in the future
            if q_quarter < current_quarter:
                continue

            # Must be within EV's time window
            if self.start_quarter != START_QUARTER_NONE:
                if q_quarter < self.start_quarter:
                    continue
            if self.ready_quarter != READY_QUARTER_NONE:
                if q_quarter > self.ready_quarter:
                    continue

            # Must be below price limit if enabled
            if self.apply_limit and q["value"] > self.max_price:
                continue

            eligible.append(i)

        return eligible


class SerialChargingScheduler:
    """Central scheduler for serial EV charging with global optimization"""

    def __init__(self, group_name: str):
        self.group_name = group_name
        self.config_entries: dict[str, EVSmartConfigEntry] = {}
        self.price_data = None
        self.last_schedule = {}  # entry_id -> [quarter_indices]
        self.last_schedule_time = None
        self.schedule_version = 0
        self.ev_schedule_version: dict[str, int] = {}
        self._lock = asyncio.Lock()

    def register(self, entry: EVSmartConfigEntry):
        self.config_entries[entry.entry_id] = entry
        self._clear_cache()

    def is_empty(self):
        return len(self.config_entries) == 0

    def unregister(self, entry: EVSmartConfigEntry):
        del self.config_entries[entry.entry_id]
        self._clear_cache()

    async def update_others(self, me: EVSmartConfigEntry):
        # Needs some kind of generational mechanism to not end up in infinite loop
        for entry_id, entry in self.config_entries.items():
            if entry_id != me.entry_id:
                _LOGGER.debug(f"[{me.title}] Calling update_sensor on {entry.title}")
                await entry.runtime_data.update_sensors(update_serial_scheduler=False)



    def _clear_cache(self):
        self.last_schedule = None
        self.schedule_version+=1
        
    async def update_price_data(self, price_data: Raw):
        """Update price data and invalidate schedule"""
        async with self._lock:
            self.price_data = price_data
            self.last_schedule = None
            self.schedule_version += 1

    async def get_schedule(self, entry: EVSmartConfigEntry, maybe_recalculate: bool = True) -> list[int]:
        """
        Get the charging schedule (list of quarter indices) for a coordinator.
        Recalculates global schedule if needed.

        Called from helpers.coordinator::create_base_schedule
        """

        # Check if we need to recalculate
        if self._needs_recalculate() and maybe_recalculate:
            async with self._lock:
                # Double-check after acquiring lock
                if self._needs_recalculate():
                    await self._calculate_global_schedule()
                    await self.update_others(entry)

        if self.last_schedule is None:
            return []

        return self.last_schedule.get(entry.entry_id, [])

    def _needs_recalculate(self) -> bool:
        """Check if schedule needs recalculation"""
        if self.last_schedule is None:
            return True
        if self.price_data is None:
            return True
        return False

    async def _calculate_global_schedule(self):
        """Calculate the globally optimal charging schedule"""
        if not self.config_entries or self.price_data is None:
            self.last_schedule = {}
            self.last_schedule_time = dt.now()
            return

        _LOGGER.debug(f"[{self.group_name}] Calculating global serial schedule")

        # Step 1: Collect EV data
        evs = self._collect_ev_data()

        _LOGGER.debug(f"evs: {evs}")

        # Step 2: Get quarter data
        quarters = self.price_data.get_raw()
        num_quarters = len(quarters)
        current_quarter = Utils.datetime_quarter(dt.now())

        # Step 3: For each EV, determine eligible quarters
        ev_eligible = {}
        for ev in evs:
            ev_eligible[ev.ev_id] = ev.get_eligible_quarters(quarters, current_quarter)

        # Step 4: Create reverse mapping: quarter -> list of eligible EV IDs
        quarter_to_evs = {i: [] for i in range(num_quarters)}
        for ev in evs:
            for q in ev_eligible[ev.ev_id]:
                quarter_to_evs[q].append(ev.ev_id)

        # Step 5: Sort quarters by price (ascending)
        price_sorted_quarters = sorted(
            range(num_quarters),
            key=lambda i: quarters[i]["value"]
        )

        # Step 6: Calculate effective priorities for each EV
        ev_priority = self._calculate_priorities(evs, ev_eligible, quarters, current_quarter)

        # Step 7: Greedy assignment
        assignment = await self._greedy_assign(
            evs, price_sorted_quarters, quarter_to_evs, ev_priority, quarters
        )

        # Step 8: Handle EVs that didn't get enough charge
        assignment = await self._handle_unmet_requirements(
            evs, assignment, quarters, quarter_to_evs, current_quarter
        )

        # Step 9: Finalize
        self.last_schedule = {}
        for ev_id in self.config_entries.keys():
            self.last_schedule[ev_id] = sorted(assignment.get(ev_id, []))

        self.last_schedule_time = dt.now()
        self.schedule_version += 1

        total_quarters = sum(len(v) for v in assignment.values())
        _LOGGER.debug(
            f"[{self.group_name}] Global schedule calculated. "
            f"Total quarters: {total_quarters}, EVs: {len(evs)}"
        )

    def _collect_ev_data(self) -> list[EVData]:
        """Collect EV data from all registered coordinators"""
        evs = []

        for ev_id, config_entry in self.config_entries.items():
            coordinator = config_entry.runtime_data
            # Skip if not active

            if not coordinator.switch_active:
                _LOGGER.debug(
                    f"[{self.group_name}] Skipping {ev_id}: smart charging not active"
                )
            if not coordinator.switch_ev_connected:
                _LOGGER.debug(
                    f"[{self.group_name}] Skipping {ev_id}: EV not connected"
                )
                # Still include with zero requirements for visibility
                evs.append(EVData(
                    ev_id=ev_id,
                    name=config_entry.title,
                    soc_current=0,
                    soc_target=0,
                    soc_min=0,
                    speed=coordinator.charging_pct_per_hour,
                    start_quarter=cast(int, coordinator.start_quarter_local),
                    ready_quarter=cast(int, coordinator.ready_quarter_local),
                    priority=coordinator.serial_charging_priority,
                    max_price=0,
                    apply_limit=False,
                    connected=False,
                    active=True,
                ))
                continue

            # Get SOC values
            soc_current = coordinator.ev_soc if coordinator.ev_soc is not None else 0
            soc_target = (
                coordinator.ev_target_soc
                if coordinator.ev_target_soc is not None
                else 100
            )
            soc_min = coordinator.number_min_soc

            ev_data = EVData(
                ev_id=ev_id,
                name=coordinator.config_entry.title,
                soc_current=soc_current,
                soc_target=soc_target,
                soc_min=soc_min,
                speed=coordinator.charging_pct_per_hour,
                start_quarter=cast(int, coordinator.start_quarter_local),
                ready_quarter=cast(int, coordinator.ready_quarter_local),
                priority=coordinator.serial_charging_priority,
                max_price=coordinator.max_price,
                apply_limit=coordinator.switch_apply_limit,
                connected=True,
                active=True,
            )

            # Only include if needs charging
            if ev_data.required > 0:
                evs.append(ev_data)
            else:
                _LOGGER.debug(
                    f"[{self.group_name}] EV {ev_id} already at target SOC"
                )

        return evs

    def _calculate_priorities(
        self, evs: list[EVData], ev_eligible: dict, quarters: list[dict], current_quarter: int
    ) -> dict:
        """Calculate effective priority score for each EV"""
        priorities = {}
        total_needed = sum(ev.required for ev in evs)

        for ev in evs:
            # Time urgency: how soon is the deadline?
            if ev.ready_quarter == READY_QUARTER_NONE:
                time_to_deadline = float('inf')
            elif ev.ready_quarter >= current_quarter:
                time_to_deadline = ev.ready_quarter - current_quarter
            else:
                time_to_deadline = 0

            urgency = min(100, max(0, 100 - time_to_deadline))

            # Need ratio
            need_ratio = ev.required / total_needed if total_needed > 0 else 0

            # SOC deficit (lower SOC = higher priority)
            soc_deficit = max(0, 100 - ev.soc_current)

            # User priority
            user_priority = ev.priority

            # Combined score (0-100 scale)
            priorities[ev.ev_id] = (
                user_priority * 0.4 +
                urgency * 0.3 +
                need_ratio * 100 * 0.2 +
                soc_deficit * 0.1
            )

        _LOGGER.debug(f"Serial Charging Priorities: {priorities}")

        return priorities

    async def _greedy_assign(
        self, evs: list[EVData], price_sorted_quarters: list[int],
        quarter_to_evs: dict, ev_priority: dict, quarters: list[dict]
    ) -> dict:
        """Greedy assignment of quarters to EVs"""
        assignment = {ev.ev_id: [] for ev in evs}
        ev_remaining = {ev.ev_id: ev.required for ev in evs}
        quarter_assigned = {i: False for i in range(len(quarters))}

        # Process quarters from cheapest to most expensive
        for q in price_sorted_quarters:
            if quarter_assigned[q]:
                continue

            # Find eligible EVs for this quarter
            candidate_evs = [
                ev_id for ev_id in quarter_to_evs[q]
                if ev_remaining.get(ev_id, 0) > 0
            ]

            if not candidate_evs:
                continue

            # Select EV with highest priority
            best_ev = max(candidate_evs, key=lambda ev_id: ev_priority[ev_id])

            # Assign this quarter to the best EV
            assignment[best_ev].append(q)
            quarter_assigned[q] = True
            ev_remaining[best_ev] -= 1

        return assignment

    async def _handle_unmet_requirements(
        self, evs: list[EVData], assignment: dict,
        quarters: list[dict], quarter_to_evs: dict, current_quarter: int
    ) -> dict:
        """
        Handle EVs that didn't get all required quarters.
        For min_SOC requirement, allow charging above price limit.
        """
        # Check which EVs didn't get enough
        for ev in evs:
            assigned_count = len(assignment[ev.ev_id])
            if assigned_count >= ev.required:
                continue  # Satisfied

            # Need to find more quarters for this EV
            missing = ev.required - assigned_count
            _LOGGER.warning(
                f"[{self.group_name}] EV {ev.name} ({ev.ev_id}) missing {missing} quarters"
            )

            # For min_SOC, we can charge above price limit
            if assigned_count >= ev.required_for_min:
                # At least min_SOC will be met, target_SOC is a nice-to-have
                _LOGGER.debug(
                    f"[{self.group_name}] EV {ev.name} will reach min_SOC but not target"
                )
                continue

            # Need to meet min_SOC - relax price limit
            _LOGGER.debug(
                f"[{self.group_name}] EV {ev.name} trying to meet min_SOC"
            )

            # Find any eligible quarters (ignoring price limit for min_SOC)
            eligible_all = ev.get_eligible_quarters(quarters, current_quarter)

            # Also include quarters that exceed price limit
            for i, q in enumerate(quarters):
                q_quarter = Utils.datetime_quarter(dt.as_local(q["start"]))

                if q_quarter < current_quarter:
                    continue
                if ev.start_quarter != START_QUARTER_NONE:
                    if q_quarter < ev.start_quarter:
                        continue
                if ev.ready_quarter != READY_QUARTER_NONE:
                    if q_quarter > ev.ready_quarter:
                        continue
                if i not in eligible_all:
                    eligible_all.append(i)

            # Sort by price and try to assign
            eligible_all.sort(key=lambda i: quarters[i]["value"])

            assigned_count_extra = 0
            for q in eligible_all:
                if assigned_count_extra >= missing:
                    break
                if quarter_to_evs[q] and q not in assignment.get(ev.ev_id, []):
                    # Check if this quarter is already assigned
                    already_assigned = False
                    for other_ev in evs:
                        if other_ev.ev_id != ev.ev_id and q in assignment.get(other_ev.ev_id, []):
                            already_assigned = True
                            break

                    if not already_assigned:
                        # Steal this quarter from the current assignment
                        can_steal = True
                        for other_ev in evs:
                            if other_ev.ev_id != ev.ev_id and q in assignment[other_ev.ev_id]:
                                # Remove from other EV
                                assignment[other_ev.ev_id].remove(q)
                                # Check if other EV still meets min_SOC
                                if len(assignment[other_ev.ev_id]) < other_ev.required_for_min:
                                    # Can't steal - other EV needs this for min_SOC
                                    assignment[other_ev.ev_id].append(q)  # Restore
                                    can_steal = False
                                    break

                        if can_steal:
                            # Assign to this EV
                            assignment[ev.ev_id].append(q)
                            assigned_count_extra += 1

            if assigned_count_extra < missing:
                _LOGGER.error(
                    f"[{self.group_name}] EV {ev.name} cannot meet min_SOC! "
                    f"Needs {missing}, could only get {assigned_count_extra}"
                )

        return assignment

@dataclass
class SerialSchedulingGroupContainer:
    _serial_schedulers: dict[str, SerialChargingScheduler] = field(default_factory=dict)

    def register(self, group_name: str, entry: EVSmartConfigEntry) -> SerialChargingScheduler:
        if not group_name in self._serial_schedulers:
            _LOGGER.debug(f"Creating SerialChargingScheduler Group {group_name}")
            self._serial_schedulers[group_name] = SerialChargingScheduler(group_name)

        _LOGGER.debug(f"Registering {entry} with SerialChargingSheduler group {group_name}")
        self._serial_schedulers[group_name].register(entry)

        return self._serial_schedulers[group_name]

    def deregister(self, group: str, entry: EVSmartConfigEntry):
        _LOGGER.debug(f"Unregistering {entry} from SerialChargingSheduler group {group}")
        self._serial_schedulers[group].unregister(entry)

        if self._serial_schedulers[group].is_empty():
            _LOGGER.debug(f"Removing empty SerialScheduler group {group}")
            del self._serial_schedulers[group]
            



