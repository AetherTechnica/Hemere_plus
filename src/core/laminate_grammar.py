from __future__ import annotations

from dataclasses import dataclass

from src.core.section_types import LaminateStack, LayerKind, LayerSpec


@dataclass(frozen=True)
class LaminateGrammarConfig:
    phi_min_deg: int = 20
    phi_max_deg: int = 180
    phi_step_deg: int = 20
    max_concentrated_run: int = 10
    max_total_cap_plies: int = 40


@dataclass(frozen=True)
class LaminateState:
    cap_phis: tuple[int, ...]
    last_phi_in_run: int
    concentrated_run_length: int

    @property
    def total_cap_plies(self) -> int:
        return len(self.cap_phis)


class LaminateGrammar:
    def __init__(self, config: LaminateGrammarConfig | None = None):
        self.config = config or LaminateGrammarConfig()
        self._validate_config()
        self.phi_values = tuple(
            phi
            for phi in range(
                self.config.phi_min_deg,
                self.config.phi_max_deg + self.config.phi_step_deg,
                self.config.phi_step_deg,
            )
            if phi <= self.config.phi_max_deg
        )
        if self.config.phi_max_deg not in self.phi_values:
            self.phi_values = tuple(sorted(set(self.phi_values + (self.config.phi_max_deg,))))

    def initial_state(self) -> LaminateState:
        return LaminateState(
            cap_phis=tuple(),
            last_phi_in_run=self.config.phi_max_deg,
            concentrated_run_length=0,
        )

    def initial_stack(self) -> LaminateStack:
        return self.build_stack(tuple())

    def next_states(self, state: LaminateState) -> tuple[LaminateState, ...]:
        if state.total_cap_plies >= self.config.max_total_cap_plies:
            return tuple()

        states: list[LaminateState] = []
        for phi in self.phi_values:
            next_state = self.append_phi(state, phi)
            if next_state is not None:
                states.append(next_state)
        return tuple(states)

    def append_phi(self, state: LaminateState, phi_deg: int) -> LaminateState | None:
        if not self._is_phi_allowed(phi_deg):
            return None
        if state.total_cap_plies >= self.config.max_total_cap_plies:
            return None

        if phi_deg == self.config.phi_max_deg:
            return LaminateState(
                cap_phis=state.cap_phis + (phi_deg,),
                last_phi_in_run=self.config.phi_max_deg,
                concentrated_run_length=0,
            )

        if state.concentrated_run_length >= self.config.max_concentrated_run:
            return None
        if phi_deg > state.last_phi_in_run:
            return None

        return LaminateState(
            cap_phis=state.cap_phis + (phi_deg,),
            last_phi_in_run=phi_deg,
            concentrated_run_length=state.concentrated_run_length + 1,
        )

    def is_valid_cap_sequence(self, cap_phis: tuple[int, ...] | list[int]) -> bool:
        state = self.initial_state()
        for phi in cap_phis:
            next_state = self.append_phi(state, int(phi))
            if next_state is None:
                return False
            state = next_state
        return True

    def build_stack(self, cap_phis: tuple[int, ...] | list[int]) -> LaminateStack:
        phis = tuple(int(phi) for phi in cap_phis)
        if not self.is_valid_cap_sequence(phis):
            raise ValueError(f"invalid cap Phi sequence: {phis}")

        layers = [
            LayerSpec(LayerKind.INNER_90, label="inner_90"),
            LayerSpec(LayerKind.BIAS_PLUS_45, label="+45"),
            LayerSpec(LayerKind.BIAS_MINUS_45, label="-45"),
        ]
        layers.extend(
            LayerSpec(LayerKind.CAP_0, phi_deg=float(phi), label=f"Phi{phi}")
            for phi in phis
        )
        layers.append(LayerSpec(LayerKind.OUTER_90, label="outer_90"))
        return LaminateStack(tuple(layers))

    def _is_phi_allowed(self, phi_deg: int) -> bool:
        cfg = self.config
        if phi_deg < cfg.phi_min_deg or phi_deg > cfg.phi_max_deg:
            return False
        return (phi_deg - cfg.phi_min_deg) % cfg.phi_step_deg == 0 or phi_deg == cfg.phi_max_deg

    def _validate_config(self) -> None:
        cfg = self.config
        if cfg.phi_min_deg <= 0:
            raise ValueError("phi_min_deg must be positive")
        if cfg.phi_max_deg != 180:
            raise ValueError("phi_max_deg must be 180 for the current HPA grammar")
        if cfg.phi_min_deg >= cfg.phi_max_deg:
            raise ValueError("phi_min_deg must be smaller than phi_max_deg")
        if cfg.phi_step_deg <= 0:
            raise ValueError("phi_step_deg must be positive")
        if cfg.max_concentrated_run <= 0:
            raise ValueError("max_concentrated_run must be positive")
        if cfg.max_total_cap_plies < 0:
            raise ValueError("max_total_cap_plies must be non-negative")
