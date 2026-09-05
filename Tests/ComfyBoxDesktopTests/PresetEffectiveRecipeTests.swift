// PresetEffectiveRecipeTests.swift — the "effective recipe" view model (#277)

import Testing
import Foundation
import ZImage
@testable import ComfyBoxDesktop

@Suite("PresetEffectiveRecipePresenter")
struct PresetEffectiveRecipeTests {

    /// The live `Krea-Kira` record from issue #277 — expandable (declares
    /// both `model` and `checkpoint_family`). Todd 2026-09-04: kroma is a
    /// regular LoRA now, declared directly in `loras[]` (the canonical form
    /// post-migration) rather than a separate structured field.
    private static let kreaKira = ImagePreset(
        id: "krea-kira", name: "Krea-Kira",
        mediaKind: "image", provider: "local", engine: "zimage",
        model: "krea2-raw",
        steps: 12, guidance: 1.0,
        width: 1024, height: 1536,
        loras: [
            LoraReference(filename: "krea2_turbo_lora_rank_64_bf16.safetensors", scale: 1.0, role: "accel"),
            LoraReference(filename: "kroma-v0.3-base-lora-rank-384-fro-0985.safetensors", scale: 0.6, role: "kroma"),
        ],
        checkpointFamily: "raw-accel",
        sampler: "res_2s", sigmaSchedule: "beta57", shift: 1.15, eta: 0.5
    )

    @Test("resolves an expandable preset: declared loras[] order, sampler/schedule/shift/eta carried through")
    func expandablePresetResolves() {
        let recipe = PresetEffectiveRecipePresenter.compute(declared: Self.kreaKira)
        #expect(recipe.unresolved == nil)
        #expect(recipe.model == "krea2-raw")
        #expect(recipe.checkpointFamily == "raw-accel")
        #expect(recipe.steps == 12)
        #expect(recipe.guidance == 1.0)
        #expect(recipe.sampler == "res_2s")
        #expect(recipe.sigmaSchedule == "beta57")
        #expect(recipe.shift == 1.15)
        #expect(recipe.eta == 0.5)
        // Todd 2026-09-04: no prepend, no special ordering — the stack is
        // `loras[]` exactly as declared.
        #expect(recipe.loraStack.count == 2)
        #expect(recipe.loraStack[0].role == "accel")
        #expect(recipe.loraStack[1].role == "kroma")
        #expect(recipe.loraStack[1].filename == "kroma-v0.3-base-lora-rank-384-fro-0985.safetensors")
        #expect(recipe.loraStack[1].scale == 0.6)
    }

    @Test("no model and no checkpoint_family: unresolved no_model, with the #359 hint")
    func noModelIsUnresolvedWithHint() {
        let preset = ImagePreset(
            id: "desktop-saved", name: "Desktop Saved", engine: "zimage",
            customModelPath: "/Models/some-checkpoint.safetensors")
        let recipe = PresetEffectiveRecipePresenter.compute(declared: preset)
        #expect(recipe.unresolved?.code == "no_model")
        // Fix round 1: the hint must point at `model`. Telling the user to
        // add `checkpoint_family` here was wrong — `decide` never reads it
        // for a `{preset}`-only request.
        #expect(recipe.unresolved?.hint == PresetEffectiveRecipePresenter.noModelHint)
        #expect(recipe.unresolved?.hint?.contains("model") == true)
        #expect(recipe.loraStack.isEmpty)
    }

    @Test("declaring model alone (no checkpoint_family) is enough to be expandable")
    func modelAloneIsExpandable() {
        let preset = ImagePreset(
            id: "model-only", name: "Model Only", engine: "zimage", model: "krea2-raw",
            loras: [LoraReference(filename: "style.safetensors", scale: 0.7)])
        let recipe = PresetEffectiveRecipePresenter.compute(declared: preset)
        #expect(recipe.unresolved == nil)
        #expect(recipe.model == "krea2-raw")
        #expect(recipe.loraStack.map(\.filename) == ["style.safetensors"])
    }

    /// `PresetLoRAStack.decide` only consults `checkpoint_family` to bridge a
    /// preset that names no model onto a REQUEST's own model (#286 round 2).
    /// The effective-recipe panel has no request context (there is no
    /// accompanying render), so a preset that declares `checkpoint_family`
    /// but not `model` still resolves `no_model` here — matching the real
    /// `POST /v1/generate {"preset": id}` shape the daemon actually sends
    /// with no other fields, not a false "this alone is enough" claim.
    @Test("checkpoint_family alone, without model, is still no_model (there is no request model to bridge to)")
    func checkpointFamilyAloneWithoutModelStillUnresolved() {
        let preset = ImagePreset(id: "family-only", name: "Family Only", engine: "zimage", checkpointFamily: "raw-accel")
        let recipe = PresetEffectiveRecipePresenter.compute(declared: preset)
        #expect(recipe.unresolved?.code == "no_model")
        #expect(recipe.unresolved?.hint == PresetEffectiveRecipePresenter.noModelHint)
    }

    /// Todd 2026-09-04: the `kroma_file_missing` gate is retired along with
    /// kroma's special semantics — a structured `kroma` field (deprecated,
    /// derived) never gates expansion; only `loras[]` matters.
    @Test("a deprecated structured kroma field never gates expansion")
    func deprecatedKromaFieldDoesNotGateExpansion() {
        let preset = ImagePreset(
            id: "krea2-x", name: "X", engine: "zimage", model: "krea2-raw",
            loras: [LoraReference(filename: "a.safetensors", scale: 0.5)],
            kroma: KromaPolicy(strength: 0.5, file: nil))
        let recipe = PresetEffectiveRecipePresenter.compute(declared: preset)
        #expect(recipe.unresolved == nil)
        #expect(recipe.loraStack.map(\.filename) == ["a.safetensors"])
    }

    @Test("a non-local engine is unresolved, never expanded onto this engine")
    func nonLocalEngineIsUnresolved() {
        let preset = ImagePreset(id: "mflux-preset", name: "Mflux", engine: "mflux", model: "schnell")
        let recipe = PresetEffectiveRecipePresenter.compute(declared: preset)
        #expect(recipe.unresolved?.code == "engine:mflux")
    }

    /// The instruction-parity check: feed the presenter the shape
    /// `POST /v1/presets/resolve` actually returns (snake_case, decoded with
    /// `ResolvedPreset`'s own Codable conformance) rather than building a
    /// `ResolvedPreset` by hand — this is what a network-driven refresh of
    /// the panel would decode.
    @Test("computes from a decoded /v1/presets/resolve response fixture")
    func computesFromResolveResponseFixture() throws {
        let json = #"""
        {
          "id": "krea-kira", "name": "Krea-Kira", "description": "",
          "media_kind": "image", "provider": "local", "engine": "zimage",
          "model": "krea2-raw", "checkpoint_family": "raw-accel",
          "steps": 12, "guidance": 1.0, "width": 1024, "height": 1536,
          "loras": [
            {"filename": "krea2_turbo_lora_rank_64_bf16.safetensors", "scale": 1.0, "role": "accel"},
            {"filename": "kroma-v0.3-base-lora-rank-384-fro-0985.safetensors", "scale": 0.6, "role": "kroma"}
          ],
          "kroma": {"strength": 0.6, "file": "kroma-v0.3-base-lora-rank-384-fro-0985.safetensors"},
          "kroma_deprecated": true,
          "sampler": "res_2s", "sigma_schedule": "beta57", "shift": 1.15, "eta": 0.5,
          "injected_keywords": []
        }
        """#
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let resolved = try decoder.decode(ResolvedPreset.self, from: Data(json.utf8))
        #expect(resolved.checkpointFamily == "raw-accel")
        #expect(resolved.sigmaSchedule == "beta57")
        #expect(resolved.kromaDeprecated == true)

        let recipe = PresetEffectiveRecipePresenter.compute(resolved: resolved, declared: Self.kreaKira)
        #expect(recipe.unresolved == nil)
        // Todd 2026-09-04: declared loras[] order, not kroma-first.
        #expect(recipe.loraStack.map(\.role) == ["accel", "kroma"])
        #expect(recipe.sigmaSchedule == "beta57")
    }

    @Test("an unsaved (empty-id) preset still resolves, so a new preset's panel is never blank")
    func unsavedPresetStillResolves() {
        let preset = ImagePreset(
            id: "", name: "Draft", engine: "zimage", model: "krea2-raw",
            checkpointFamily: "raw-accel")
        let recipe = PresetEffectiveRecipePresenter.compute(declared: preset)
        #expect(recipe.unresolved == nil)
        #expect(recipe.model == "krea2-raw")
    }
}
