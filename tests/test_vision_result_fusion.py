import unittest

from services.vision.result_fusion import fuse_multimodal_results
from services.vision.schemas import (
    OCRBlock,
    OCRExtraction,
    OCRMeasurement,
    VisionInferenceResult,
    VisualConnection,
    VisualExtraction,
)


def _ocr(text: str, measurements=None, discarded_blocks: int = 0) -> OCRExtraction:
    return OCRExtraction(
        engine="paddleocr-cpu",
        detection_model="det",
        recognition_model="rec",
        image_path="ocr.png",
        full_text=text,
        blocks=[OCRBlock(text=text, confidence=0.99, bbox=[])],
        measurements=list(measurements or []),
        discarded_blocks=discarded_blocks,
        model_load_ms=1.0,
        inference_ms=2.0,
    )


def _vision(extraction: VisualExtraction, warnings=()) -> VisionInferenceResult:
    return VisionInferenceResult(
        model="qwen3-vl:2b-instruct",
        prompt_version="visual-extraction-v1",
        image_path="vlm.png",
        extraction=extraction,
        normalization_warnings=tuple(warnings),
        wall_seconds=1.0,
        total_seconds=1.0,
        load_seconds=0.0,
        prompt_tokens=100,
        generated_tokens=50,
        tokens_per_second=50.0,
    )


class VisionResultFusionTests(unittest.TestCase):
    def test_exact_ocr_measurements_win_and_external_brand_blocks_retrieval(self):
        amp = OCRMeasurement("180ADC", 180.0, "A", "DC", "180 A DC")
        ocr = _ocr(
            "Guangzhou Anern Energy Technology Co., Ltd. AC Grid 32A Breaker "
            "Inverter1 180ADC",
            [amp],
            discarded_blocks=1,
        )
        visual = VisualExtraction(
            image_type="wiring_diagram",
            manufacturer="Guangzhou Anern Energy Technology Co., Ltd.",
            product_model=None,
            visible_status=[],
            connections=[
                VisualConnection(source="AC Grid", target="32A Breaker", label="32A"),
                VisualConnection(source="Battery", target="Unknown Port", label=None),
            ],
            observations=["A wiring diagram is visible."],
            possible_faults=[],
            out_of_domain=False,
            confidence=0.9,
            uncertainties=[],
        )

        result = fuse_multimodal_results(ocr, _vision(visual))

        self.assertTrue(result.out_of_domain)
        self.assertFalse(result.retrieval_allowed)
        self.assertEqual(result.retrieval_query, "")
        self.assertEqual(result.measurements, (amp,))
        self.assertTrue(result.connections[0].ocr_terms_present)
        self.assertFalse(result.connections[1].ocr_terms_present)
        self.assertIn("OCR discarded 1", " ".join(result.warnings))

    def test_ocr_domain_brand_and_model_enable_retrieval(self):
        ocr = _ocr("HANCHU ESS HESS-HY-T-15K Fault E101 230V")
        visual = VisualExtraction(
            image_type="lcd_screen",
            manufacturer="Hanchu ESS",
            product_model="HESS-HY-T-15K",
            visible_status=["Fault E101", "Fault E101"],
            connections=[],
            observations=["The display shows E101."],
            possible_faults=["Fault E101 is displayed."],
            out_of_domain=True,
            confidence=0.95,
            uncertainties=[],
        )

        result = fuse_multimodal_results(
            ocr,
            _vision(visual),
            user_question="What does this fault mean?",
        )

        self.assertEqual(result.manufacturer, "Hanchu")
        self.assertEqual(result.product_model, "HESS-HY-T-15K")
        self.assertFalse(result.out_of_domain)
        self.assertTrue(result.retrieval_allowed)
        self.assertEqual(result.visible_status, ("Fault E101",))
        self.assertEqual(result.possible_faults, ("Fault E101 is displayed.",))
        self.assertIn("What does this fault mean?", result.retrieval_query)

    def test_unsupported_model_and_fault_are_removed(self):
        ocr = _ocr("EnValt inverter 230V")
        visual = VisualExtraction(
            image_type="nameplate",
            manufacturer="EnValt",
            product_model="HESS-FAKE-99",
            visible_status=[],
            connections=[],
            observations=[],
            possible_faults=["Fault E999 is displayed."],
            out_of_domain=False,
            confidence=0.8,
            uncertainties=[],
        )

        result = fuse_multimodal_results(ocr, _vision(visual))

        self.assertIsNone(result.product_model)
        self.assertEqual(result.possible_faults, ())
        self.assertEqual(len(result.conflicts), 2)
        self.assertEqual(result.to_dict()["manufacturer"], "EnValt")


if __name__ == "__main__":
    unittest.main()
