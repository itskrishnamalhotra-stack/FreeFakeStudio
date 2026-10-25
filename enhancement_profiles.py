"""Editable, versioned prompt library. Routing uses IDs, never inferred prose."""
VERSION = "scene-profiles-v1"

PRESERVATION = (
    "Image 1 is the authoritative source. Enhance its clarity in its original medium. "
    "Keep identity, apparent age, facial geometry, body proportions, expression, pose, clothes, "
    "object count, composition, perspective, text, exposure and colors unchanged. "
    "Restore supported fine edge separation and believable detail rather than merely changing brightness. "
    "Correct only clear compression or rendering defects whose intended structure is unambiguous. "
    "Do not invent freckles, spots, wrinkles, coarse pores, body hair, patterns or objects. "
    "Preserve the source's existing complexion and texture scale. Preserve intentional blur, "
    "motion and depth of field. Do not beautify, reshape, relight or redesign the image. "
)

PROFILES = {
    "female_portrait": {
        "name": "Female-Presenting Portrait",
        "description": "Facial detail and hair, preserving identity and complexion.",
        "prompt": "Refine the main subject's in-focus eyes, facial contours, eyebrows, existing hair strands and clothing edges. Preserve her exact facial structure, expression, complexion and existing skin characteristics. Do not add makeup, freckles, wrinkles or heavy pores, reshape the face or change apparent age.",
    },
    "male_portrait": {
        "name": "Male-Presenting Portrait",
        "description": "Facial detail and existing facial hair, without inventing ruggedness.",
        "prompt": "Refine the main subject's in-focus eyes, facial contours, eyebrows, existing hair and facial hair where present. Preserve his exact facial structure, expression and complexion. Do not invent beard growth, deepen wrinkles, strengthen the jaw or make the face more muscular or rugged.",
    },
    "female_body": {
        "name": "Female-Presenting Body / Fashion",
        "description": "Clothing and visible body detail, preserving proportions.",
        "prompt": "Improve the clarity of the visible person, garment construction, existing fabric weave, seams, folds and hair detail. Preserve her exact body shape, proportions, posture and anatomy. Keep detail appropriate to camera distance. Do not slim, enlarge, reshape, beautify or add body definition.",
    },
    "male_body": {
        "name": "Male-Presenting Body / Fashion",
        "description": "Clothing and visible body detail, without adding muscles or definition.",
        "prompt": "Improve the clarity of the visible person, garment construction, existing fabric weave, seams, folds and hair detail. Preserve his exact body shape, proportions, posture and anatomy. Keep detail appropriate to camera distance. Do not add muscles, veins, body hair or stronger anatomical definition.",
    },
    "group": {
        "name": "Couples / Groups",
        "description": "Multiple people, keeping their identities and features distinct.",
        "prompt": "Enhance each visible person independently while preserving their distinct identities, appearances, expressions and positions. Keep the exact number of people and their interactions unchanged. Do not transfer facial features, clothing or texture between subjects. Match detail to each person's size and focus. Keep incidental background people soft.",
    },
    "person_unspecified": {
        "name": "Unclear Presentation / Obscured Person",
        "description": "Visible details only, without inventing hidden features.",
        "prompt": "Preserve the visible person exactly as shown. Refine supported in-focus facial details only where actually visible, clothing, hair, silhouette and anatomical edges. Do not infer gender, invent hidden facial features, reveal obscured areas or turn the person toward the camera. Preserve the same body shape and proportions without idealization.",
    },
    "landscape": {
        "name": "Landscapes / Nature",
        "description": "Terrain, vegetation and water, preserving atmosphere and distance.",
        "prompt": "Clarify supported in-focus terrain, leaf shapes, rocks and water at their existing scale. Preserve the sky, weather, atmosphere, reflections, wave scale and distant haze. Keep distant detail naturally less distinct. Do not turn foliage into repeating patterns or add dramatic lighting.",
    },
    "architecture": {
        "name": "Architecture / Streets / Interiors",
        "description": "Structural detail, perspective and existing layout.",
        "prompt": "Refine supported architectural edges, structural details and visible surface finishes. Preserve perspective, straight lines, room layout, signs, furniture and object placement. Do not invent windows, alter geometry or change the interior design. Preserve existing reflections and material finish.",
    },
    "product": {
        "name": "Products / Vehicles / Objects",
        "description": "Shape, finish, branding and construction details.",
        "prompt": "Improve existing contours, construction details, surface finish and material separation. Preserve exact shape, branding, lettering, controls, colors and reflections. Do not invent specifications, repair intentional wear or redesign the object.",
    },
    "animal": {
        "name": "Animals",
        "description": "Existing fur, feathers or scales, preserving anatomy and markings.",
        "prompt": "Refine in-focus eyes and existing fur, feathers or scales while preserving the animal's anatomy, expression, markings and coat direction. Keep detail appropriate to distance. Do not apply human skin texture or alter the animal's breed or appearance.",
    },
    "food": {
        "name": "Food / Still Life",
        "description": "Existing ingredients, arrangement and surface detail.",
        "prompt": "Improve supported ingredient boundaries, surface texture and material separation while preserving the arrangement, portions, colors and existing moisture or gloss. Do not add ingredients, steam, droplets or artificial shine. Keep the same food shapes and existing objects.",
    },
    "macro": {
        "name": "Macro / Close-Up",
        "description": "Fine detail within the original plane of focus.",
        "prompt": "Resolve supported detail within the existing plane of focus. Preserve magnification, surface structure and shallow depth of field. Do not sharpen intentionally blurred regions or invent microscopic structures. Keep detail physically plausible at the visible scale.",
    },
    "artwork": {
        "name": "Illustration / Anime / Painting / CGI",
        "description": "Clarity within the original artistic style.",
        "prompt": "Enhance clarity within the original artistic medium. Preserve line work, brushwork, shading style, palette and intentional simplification. Do not convert the image into a photograph or introduce photographic pores, hair or material textures. Preserve deliberately stylized CGI rather than forcing photographic realism.",
    },
    "document": {
        "name": "Documents / Screenshots / Graphics",
        "description": "Legibility without inventing or changing information.",
        "prompt": "Improve legibility and clean compression artifacts without changing text, numbers, symbols, diagrams, typography or layout. Preserve uncertain characters rather than guessing replacements. Do not hallucinate missing information or replace exact graphic edges with photographic texture.",
    },
    "general": {
        "name": "General-Purpose Fallback",
        "description": "Conservative enhancement when classification is uncertain.",
        "prompt": "Improve supported edge separation and visible detail conservatively. Where the subject, material or medium is uncertain, preserve the original appearance instead of inventing detail. Avoid aggressive sharpening, texture generation and stylistic changes.",
    },
}

FOCUS_RULES = {
    "in_focus": "Prioritize the main subject's existing focal plane; preserve background softness.",
    "soft_subject": "Preserve the source subject's softness or motion blur; do not fabricate a sharply resolved identity or unsupported detail.",
    "background_people": "Treat background people as incidental scene elements. Preserve their size and softness; do not reconstruct detailed faces or skin. Their presence must not turn the scene into a portrait.",
    "multiple_subjects": "Maintain the existing relative sharpness of all subjects without making every plane equally sharp.",
    "uncertain": "Keep uncertain details understated. Do not present invented detail as recovered information.",
}


def select(analysis):
    """Never apply a gendered profile to incidental, obscured or ambiguous people."""
    scene = analysis.get("scene")
    if scene in {"artwork", "document"}:
        return scene
    if analysis.get("people_role") == "main_subject":
        if analysis.get("people_count") == "multiple":
            return "group"
        framing = analysis.get("framing")
        presentation = analysis.get("presentation")
        if analysis.get("people_count") == "single" and framing in {"portrait", "body"}:
            if presentation in {"female_presenting", "male_presenting"}:
                return presentation.split("_")[0] + ("_body" if framing == "body" else "_portrait")
        return "person_unspecified"
    if analysis.get("macro") is True:
        return "macro"
    return scene if scene in {"landscape", "architecture", "product", "animal", "food"} else "general"


def focus_rules(analysis):
    keys = []
    if analysis.get("people_role") == "background":
        keys.append("background_people")
    if analysis.get("people_count") == "multiple" and analysis.get("people_role") == "main_subject":
        keys.append("multiple_subjects")
    keys.append(analysis.get("focus") if analysis.get("focus") in {"in_focus", "soft_subject"} else "uncertain")
    return keys
