#!/usr/bin/env python3
"""EC-DynaMech executable operator engine v1.0.

Operators are implemented in two explicitly separated modes:
1. transform: applies a controlled state transition;
2. measure: derives an operator profile from measured objects/states.

Canonical boundary:
    pixels are carriers, geometry is the measurement object,
    operators are derived measurement layers.

Dependencies: numpy
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

VERSION = "1.0.0"
EPS = 1e-12


@dataclass
class ECState:
    """Mechanical/geometric state S(t)=[V,E,X,A,W,Gamma,q]."""
    state_id: str
    positions: np.ndarray                 # (n,d)
    edges: np.ndarray = field(default_factory=lambda: np.empty((0, 2), int))
    weights: Optional[np.ndarray] = None  # per edge
    velocities: Optional[np.ndarray] = None
    time: float = 0.0
    quality: float = 1.0
    sigma: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=float)
        self.edges = np.asarray(self.edges, dtype=int).reshape(-1, 2)
        if self.positions.ndim != 2 or len(self.positions) == 0:
            raise ValueError("positions moet een niet-lege matrix (n,d) zijn")
        if self.weights is None:
            self.weights = np.ones(len(self.edges), float)
        else:
            self.weights = np.asarray(self.weights, dtype=float)
        if len(self.weights) != len(self.edges):
            raise ValueError("weights en edges hebben verschillende lengte")
        if self.velocities is None:
            self.velocities = np.zeros_like(self.positions)
        else:
            self.velocities = np.asarray(self.velocities, dtype=float)
            if self.velocities.shape != self.positions.shape:
                raise ValueError("velocities moet dezelfde vorm hebben als positions")
        self.quality = float(np.clip(self.quality, 0, 1))
        self.sigma = float(max(0, self.sigma))

    def copy(self, state_id: Optional[str] = None) -> "ECState":
        return ECState(
            state_id or self.state_id,
            self.positions.copy(), self.edges.copy(), self.weights.copy(),
            self.velocities.copy(), self.time, self.quality, self.sigma,
            dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class OperatorResult:
    operator_id: str
    symbol: str
    mode: str
    status: str
    raw: dict[str, Any]
    score: Any
    unit: str
    quality: float
    sigma: float
    falsified: bool
    falsification_reason: str
    source_trace: list[str]
    output_state: Optional[ECState] = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


class RhythmOperator:
    operator_id = "EC-OP-RHO"
    symbol = "O_rho"

    def transform(self, state: ECState, amplitude: float, wavelength: float,
                  axis: int = 1, phase: float = 0.0) -> OperatorResult:
        if wavelength <= 0:
            raise ValueError("wavelength moet positief zijn")
        out = state.copy(state.state_id + ":rho")
        base_axis = 0 if axis != 0 else min(1, out.positions.shape[1] - 1)
        coordinate = out.positions[:, base_axis]
        out.positions[:, axis] += amplitude * np.sin(2 * math.pi * coordinate / wavelength + phase)
        out.metadata.update({"operator": self.symbol, "amplitude": amplitude,
                             "wavelength": wavelength, "phase": phase})
        return OperatorResult(self.operator_id, self.symbol, "transform", "APPLIED",
                              {"amplitude": amplitude, "wavelength": wavelength, "phase": phase},
                              None, "coordinate", state.quality, state.sigma, False, "",
                              [state.state_id], out)

    def measure(self, values: Sequence[float], quality: float = 1.0,
                sigma_measure: float = 0.0, bootstrap: int = 200,
                seed: int = 17) -> OperatorResult:
        x = np.asarray(values, float)
        x = x[np.isfinite(x)]
        if len(x) < 3 or np.mean(np.abs(x)) <= EPS:
            return _not_measured(self, "minimaal drie geldige niet-nul waarden vereist")
        mean = float(np.mean(x)); std = float(np.std(x, ddof=1))
        cv = abs(std / mean)
        score = float(math.exp(-cv))
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(max(20, bootstrap)):
            b = rng.choice(x, len(x), replace=True)
            m = np.mean(b)
            if abs(m) > EPS:
                boots.append(math.exp(-abs(np.std(b, ddof=1) / m)))
        sigma = float(math.sqrt((np.std(boots) if boots else 1.0) ** 2 + sigma_measure ** 2))
        q = float(np.clip(quality * min(1, len(x) / 12) * (1 - min(sigma, 1)), 0, 1))
        falsified = q >= 0.70 and score <= 0.20
        return OperatorResult(self.operator_id, self.symbol, "measure", "MEASURED",
                              {"n": len(x), "mean": mean, "std": std, "cv": cv}, score,
                              "0-1", q, sigma, falsified,
                              "regelmaat onder vooraf ingestelde ondergrens" if falsified else "",
                              ["sequence:values"])


class ModulationOperator:
    operator_id = "EC-OP-MU"
    symbol = "O_mu"

    def transform(self, state: ECState, scale_start: float, scale_end: float,
                  center: Optional[Sequence[float]] = None) -> OperatorResult:
        if scale_start <= 0 or scale_end <= 0:
            raise ValueError("schalen moeten positief zijn")
        out = state.copy(state.state_id + ":mu")
        c = np.asarray(center if center is not None else np.mean(out.positions, axis=0), float)
        progression = np.linspace(scale_start, scale_end, len(out.positions))[:, None]
        out.positions = c + (out.positions - c) * progression
        out.metadata.update({"operator": self.symbol, "scale_start": scale_start, "scale_end": scale_end})
        return OperatorResult(self.operator_id, self.symbol, "transform", "APPLIED",
                              {"scale_start": scale_start, "scale_end": scale_end},
                              float(math.log(scale_end / scale_start)), "log-ratio",
                              state.quality, state.sigma, False, "", [state.state_id], out)

    def measure(self, before: Sequence[float], after: Sequence[float],
                quality: float = 1.0, sigma_measure: float = 0.0) -> OperatorResult:
        a = np.asarray(before, float); b = np.asarray(after, float)
        if a.shape != b.shape or a.size == 0:
            return _not_measured(self, "before en after moeten dezelfde niet-lege vorm hebben")
        valid = np.isfinite(a) & np.isfinite(b) & (np.abs(a) > EPS) & (np.abs(b) > EPS)
        if not np.any(valid):
            return _not_measured(self, "geen geldige positieve verhoudingen")
        local = np.log(np.abs(b[valid]) / np.abs(a[valid]))
        raw_score = float(np.sqrt(np.mean(local ** 2)))
        score = float(1 - math.exp(-raw_score))
        sigma = float(math.sqrt((np.std(local, ddof=1) / math.sqrt(len(local)) if len(local)>1 else 0) ** 2 + sigma_measure ** 2))
        q = float(np.clip(quality * min(1, len(local)/8) * (1-min(sigma,1)),0,1))
        falsified = q >= .70 and score <= .05
        return OperatorResult(self.operator_id, self.symbol, "measure", "MEASURED",
                              {"n": len(local), "log_ratios": local, "rms_log_change": raw_score},
                              score, "0-1", q, sigma, falsified,
                              "verandering niet boven drempel" if falsified else "", ["before","after"])


class InterferenceOperator:
    def __init__(self, positive: bool) -> None:
        self.positive = positive
        self.operator_id = "EC-OP-I+" if positive else "EC-OP-I-"
        self.symbol = "O_iota+" if positive else "O_iota-"

    def transform(self, component_1: Sequence[float], component_2: Sequence[float],
                  phase_shift: float = 0.0) -> OperatorResult:
        x1 = np.asarray(component_1, float); x2 = np.asarray(component_2, float)
        if x1.shape != x2.shape or x1.size == 0:
            raise ValueError("componenten moeten dezelfde niet-lege vorm hebben")
        factor = math.cos(phase_shift)
        total = x1 + factor * x2
        measured = self.measure(x1*x1, x2*x2, total*total, quality=1.0)
        measured.mode = "transform"
        measured.raw["phase_shift"] = phase_shift
        measured.raw["result"] = total
        return measured

    def measure(self, intensity_1: Sequence[float], intensity_2: Sequence[float],
                intensity_total: Sequence[float], quality: float = 1.0,
                sigma_measure: float = 0.0) -> OperatorResult:
        i1=np.asarray(intensity_1,float); i2=np.asarray(intensity_2,float); it=np.asarray(intensity_total,float)
        if i1.shape != i2.shape or i1.shape != it.shape or i1.size == 0:
            return _not_measured(self, "I, I1 en I2 moeten dezelfde niet-lege vorm hebben")
        delta = it - i1 - i2
        denom = np.abs(i1) + np.abs(i2) + EPS
        local = np.maximum(0, delta if self.positive else -delta) / denom
        score = float(np.clip(np.mean(local), 0, 1))
        sigma = float(math.sqrt((np.std(local,ddof=1)/math.sqrt(len(local)) if len(local)>1 else 0)**2 + sigma_measure**2))
        q=float(np.clip(quality*min(1,len(local)/8)*(1-min(sigma,1)),0,1))
        falsified=q>=.70 and score<=.02
        return OperatorResult(self.operator_id,self.symbol,"measure","MEASURED",
                              {"n":len(local),"interaction":delta,"local_score":local},score,"0-1",q,sigma,
                              falsified,"geen interactieterm boven onzekerheid" if falsified else "",
                              ["I1","I2","I_total"])


class CoherenceOperator:
    operator_id="EC-OP-C"; symbol="O_C"

    def transform(self, angles: Sequence[float], coupling: float = 0.5,
                  steps: int = 1) -> OperatorResult:
        a=np.asarray(angles,float).copy()
        for _ in range(max(1,steps)):
            mean_angle=math.atan2(np.mean(np.sin(a)),np.mean(np.cos(a)))
            delta=np.angle(np.exp(1j*(mean_angle-a)))
            a += coupling*delta
        result=self.measure(directions=a,quality=1.0)
        result.mode="transform"; result.raw["transformed_angles"]=a
        result.raw["coupling"]=coupling; result.raw["steps"]=steps
        return result

    def measure(self, directions: Sequence[float], edge_presence: Optional[Sequence[float]]=None,
                loop_closure: Optional[Sequence[float]]=None, modal_vectors: Optional[np.ndarray]=None,
                quality: float=1.0, sigma_measure: float=0.0) -> OperatorResult:
        a=np.asarray(directions,float); a=a[np.isfinite(a)]
        if len(a)<2: return _not_measured(self,"minimaal twee richtingen vereist")
        c_dir=float(abs(np.mean(np.exp(1j*a))))
        channels={"C_direction":c_dir}
        if edge_presence is not None:
            e=np.asarray(edge_presence,float); channels["C_connectivity"]=float(np.clip(np.mean(e),0,1))
        if loop_closure is not None:
            l=np.asarray(loop_closure,float); channels["C_closure"]=float(np.clip(np.mean(l),0,1))
        if modal_vectors is not None:
            m=np.asarray(modal_vectors,float)
            if m.ndim==2 and len(m)>1:
                n=m/(np.linalg.norm(m,axis=1,keepdims=True)+EPS)
                channels["C_modal"]=float(np.clip(np.mean(np.abs(n@n.T)[np.triu_indices(len(n),1)]),0,1))
        vals=np.asarray(list(channels.values()))
        sigma=float(math.sqrt((np.std(vals)/math.sqrt(len(vals)))**2+sigma_measure**2))
        q=float(np.clip(quality*min(1,len(a)/10)*(1-min(sigma,1)),0,1))
        falsified=q>=.70 and max(vals)<=.20
        return OperatorResult(self.operator_id,self.symbol,"measure","MEASURED",channels,channels,
                              "0-1 per channel",q,sigma,falsified,
                              "geen coherentiekanaal boven ondergrens" if falsified else "",["direction/topology/modal"])


class ProjectionOperator:
    operator_id="EC-OP-PI"; symbol="O_Pi"

    def transform(self, points_3d: np.ndarray, camera_matrix: np.ndarray,
                  rotation: Optional[np.ndarray]=None, translation: Optional[Sequence[float]]=None) -> OperatorResult:
        x=np.asarray(points_3d,float); k=np.asarray(camera_matrix,float)
        if x.ndim!=2 or x.shape[1]!=3 or k.shape!=(3,3):
            raise ValueError("points_3d=(n,3), camera_matrix=(3,3) vereist")
        r=np.eye(3) if rotation is None else np.asarray(rotation,float)
        t=np.zeros(3) if translation is None else np.asarray(translation,float)
        cam=(r@x.T).T+t
        if np.any(cam[:,2]<=EPS): raise ValueError("punten moeten positieve cameradiepte hebben")
        hom=(k@cam.T).T
        uv=hom[:,:2]/hom[:,2,None]
        return OperatorResult(self.operator_id,self.symbol,"transform","APPLIED",
                              {"points_2d":uv,"depth":cam[:,2]},None,"px",1.0,0.0,False,"",
                              ["points_3d","camera_matrix"])

    def measure(self, observed_2d: np.ndarray, predicted_2d: np.ndarray,
                reference_scale: Optional[float]=None, quality: float=1.0,
                sigma_measure: float=0.0, tolerance: float=0.05) -> OperatorResult:
        o=np.asarray(observed_2d,float); p=np.asarray(predicted_2d,float)
        if o.shape!=p.shape or o.ndim!=2 or o.shape[1]!=2 or len(o)==0:
            return _not_measured(self,"observed_2d en predicted_2d moeten (n,2) zijn")
        errors=np.linalg.norm(o-p,axis=1); rmse=float(math.sqrt(np.mean(errors**2)))
        scale=float(reference_scale or max(np.ptp(o[:,0]),np.ptp(o[:,1]),EPS))
        normalized=rmse/scale; score=float(math.exp(-normalized))
        sigma=float(math.sqrt((np.std(errors,ddof=1)/math.sqrt(len(errors))/scale if len(errors)>1 else 0)**2+sigma_measure**2))
        q=float(np.clip(quality*min(1,len(errors)/8)*(1-min(sigma,1)),0,1))
        falsified=q>=.70 and normalized>=tolerance
        return OperatorResult(self.operator_id,self.symbol,"measure","MEASURED",
                              {"n":len(errors),"errors":errors,"rmse":rmse,"normalized_error":normalized},
                              score,"0-1 fit; px error",q,sigma,falsified,
                              "reprojectiefout boven tolerantie" if falsified else "",["observed_2d","predicted_2d"])


class ECDynaMechEngine:
    def __init__(self) -> None:
        self.rho=RhythmOperator(); self.mu=ModulationOperator()
        self.iota_plus=InterferenceOperator(True); self.iota_minus=InterferenceOperator(False)
        self.coherence=CoherenceOperator(); self.projection=ProjectionOperator()

    def self_test(self) -> dict[str,Any]:
        theta=np.linspace(0,2*math.pi,12,endpoint=False)
        state=ECState("SELFTEST-S0",np.column_stack([np.cos(theta),np.sin(theta)]),
                      np.column_stack([np.arange(12),np.roll(np.arange(12),-1)]),quality=.95,sigma=.01)
        rho_t=self.rho.transform(state,.08,.7)
        rho_m=self.rho.measure(np.full(12,2.0),quality=.95)
        mu_t=self.mu.transform(state,1.0,1.4)
        mu_m=self.mu.measure(np.ones(12),np.linspace(1,1.4,12),quality=.95)
        x=np.sin(np.linspace(0,4*math.pi,64)); y=np.sin(np.linspace(0,4*math.pi,64))
        ip=self.iota_plus.transform(x,y,0.0); im=self.iota_minus.transform(x,y,math.pi)
        c0=np.linspace(-.8,.8,12); ct=self.coherence.transform(c0,.7,3)
        c_before=self.coherence.measure(c0); c_after=self.coherence.measure(ct.raw["transformed_angles"])
        pts=np.column_stack([np.cos(theta),np.sin(theta),np.full(12,5.)])
        k=np.array([[500,0,320],[0,500,240],[0,0,1]],float)
        proj=self.projection.transform(pts,k)
        meas=self.projection.measure(proj.raw["points_2d"],proj.raw["points_2d"]+.2,quality=.95,reference_scale=200)
        tests={
            "rho_regular_high": rho_m.score>.95,
            "mu_detects_change": mu_m.score>0,
            "iota_positive": ip.score>0,
            "iota_negative": im.score>0,
            "coherence_increases": c_after.score["C_direction"]>c_before.score["C_direction"],
            "projection_low_error": meas.raw["normalized_error"]<.01,
            "state_shapes_preserved": rho_t.output_state.positions.shape==state.positions.shape==mu_t.output_state.positions.shape,
        }
        return {"version":VERSION,"passed":all(tests.values()),"tests":tests,"results":{
            "rho_measure":rho_m.to_dict(),"mu_measure":mu_m.to_dict(),
            "iota_plus":ip.to_dict(),"iota_minus":im.to_dict(),
            "coherence_before":c_before.to_dict(),"coherence_after":c_after.to_dict(),
            "projection":meas.to_dict()}}


def _not_measured(operator: Any, reason: str) -> OperatorResult:
    return OperatorResult(operator.operator_id,operator.symbol,"measure","NOT_MEASURED",{},None,"",0.0,1.0,False,"",[],notes=reason)


def _json_safe(v: Any) -> Any:
    if isinstance(v,np.ndarray): return v.tolist()
    if isinstance(v,np.generic): return v.item()
    if isinstance(v,dict): return {str(k):_json_safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [_json_safe(x) for x in v]
    return v


def main(argv: Optional[Sequence[str]]=None) -> int:
    p=argparse.ArgumentParser(description="EC-DynaMech executable operator engine")
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--output",default="ec_operator_selftest.json")
    args=p.parse_args(argv)
    if not args.self_test: p.error("gebruik --self-test; importeer de module voor analyses")
    result=ECDynaMechEngine().self_test()
    Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result["passed"] else 2

if __name__=="__main__":
    raise SystemExit(main())
