"""Sampling-contract tests using fake ComfyUI nodes and tiny CPU tensors only."""
import sys
import types
import unittest
from unittest.mock import Mock, patch

import torch
from PIL import Image

import engine_flux_klein_4b as flux


class NativeReconstructionTests(unittest.TestCase):
    def test_uses_flux2_latent_full_schedule_and_reference_conditioning(self):
        latent = {"samples": torch.zeros(1, 128, 4, 4)}
        sigmas = torch.tensor([1., .8, .5, .2, 0.])
        samples = {"samples": "sampled"}
        empty = Mock(execute=Mock(return_value=(latent,)))
        scheduler = Mock(execute=Mock(return_value=(sigmas,)))
        custom = Mock(execute=Mock(return_value=(samples,)))
        comfy = types.ModuleType("comfy")
        comfy.samplers = types.ModuleType("comfy.samplers")
        comfy.samplers.sampler_object = Mock(return_value="euler-object")
        extras = types.ModuleType("comfy_extras")
        nodes_flux = types.ModuleType("comfy_extras.nodes_flux")
        nodes_flux.EmptyFlux2LatentImage, nodes_flux.Flux2Scheduler = empty, scheduler
        nodes_custom = types.ModuleType("comfy_extras.nodes_custom_sampler")
        nodes_custom.SamplerCustom = custom
        encoder = Mock(encode=Mock(side_effect=[("positive",), ("negative",)]))
        decoder = Mock(decode=Mock(return_value=(torch.zeros(1, 64, 64, 3),)))
        nodes = {"CLIPTextEncode": encoder, "VAEDecode": decoder}
        image = Image.new("RGB", (64, 64), "blue")
        with patch.dict(sys.modules, {"comfy": comfy, "comfy.samplers": comfy.samplers,
                    "comfy_extras": extras, "comfy_extras.nodes_flux": nodes_flux,
                    "comfy_extras.nodes_custom_sampler": nodes_custom}), \
             patch.object(flux, "_get_nodes", return_value=nodes), \
             patch.object(flux, "_add_reference_conditioning", return_value="with-reference") as reference, \
             patch.object(flux, "_unet", "model"):
            result = flux.reconstruct_reference([image], "Reconstruct this photograph", 7, 4)
        empty.execute.assert_called_once_with(64, 64, 1)
        scheduler.execute.assert_called_once_with(4, 64, 64)
        call = custom.execute.call_args.args
        self.assertEqual(call[:7], ("model", True, 7, 1., "with-reference", "negative", "euler-object"))
        self.assertIs(call[7], sigmas)
        self.assertIs(call[8], latent)
        self.assertEqual(reference.call_args.args[2], [image])
        self.assertEqual(result.size, (64, 64))
        self.assertNotIn("KSampler", nodes)
        self.assertNotIn("VAEEncode", nodes)


if __name__ == "__main__":
    unittest.main()
