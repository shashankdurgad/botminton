"""Spawner that frees the Kuka-Allegro's root so a mobile base can carry it."""

from __future__ import annotations


BASE_MASS = 1.0e4
"""Mass given to the arm's base link when it rides the mobile base [kg]."""
BASE_INERTIA = 1.0e3
"""Principal inertia of the base link [kg*m^2]."""


def spawn_floating_root_usd(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    """Spawn the robot USD, then move its articulation root from the world joint to the base link.

    The Kuka asset roots its articulation on ``root_joint``, a fixed joint from the world to ``iiwa7_link_0``.
    Disabling that joint alone leaves PhysX without a valid root, so the root API and its PhysX settings are
    moved onto the link the joint held and the joint is deactivated. That link is then made heavy and
    gravity-free, see below.
    """
    # imported here: the task config is loaded before Kit has set up its USD libraries
    from pxr import Gf, Sdf, Usd, UsdPhysics

    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd

    prim = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    for joint_prim in [p for p in Usd.PrimRange(prim) if p.IsA(UsdPhysics.FixedJoint)]:
        if not joint_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            continue
        joint = UsdPhysics.Joint(joint_prim)
        if joint.GetBody0Rel().GetTargets():
            continue  # not a world joint
        (body_path,) = joint.GetBody1Rel().GetTargets()
        body = prim.GetStage().GetPrimAtPath(body_path)
        UsdPhysics.ArticulationRootAPI.Apply(body)
        for schema in joint_prim.GetAppliedSchemas():
            if schema.startswith("Physx") and "Articulation" in schema:
                body.AddAppliedSchema(schema)
        for attr in joint_prim.GetAttributes():
            if attr.GetName().startswith("physxArticulation:") and attr.HasAuthoredValue():
                body.CreateAttribute(attr.GetName(), attr.GetTypeName()).Set(attr.Get())
        joint_prim.SetActive(False)
        # the env re-poses this link every physics step; within a step it must act like a heavy, level base:
        # no gravity of its own (or the whole arm free-falls with it and loses its gravity load) and enough
        # mass that the arm's swing reactions cannot move it
        mass = UsdPhysics.MassAPI.Apply(body)
        mass.CreateMassAttr().Set(BASE_MASS)
        mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(BASE_INERTIA, BASE_INERTIA, BASE_INERTIA))
        body.AddAppliedSchema("PhysxRigidBodyAPI")
        body.CreateAttribute("physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool).Set(True)
    return prim
