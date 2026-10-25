"""Editable, versioned prompt library. Routing uses IDs, never inferred prose."""
VERSION = "scene-profiles-v2"
PROMPT_VERSION = "user-refinement-v3"

PROFILES = {
    "female_portrait": {
        "name": "Female-Presenting Portrait",
        "description": "Facial detail and hair, preserving identity and complexion.",
        "prompt": (
            "Make this AI-generated image look like a natural, high-quality photograph. Preserve the exact person, "
            "facial features, age, complexion, expression, hairstyle, pose, lighting, and background. Gently "
            "correct obvious AI-rendering artifacts and improve natural skin, hair, eyes, and photographic detail. "
            "Keep the original skin clear, smooth, and healthy-looking, with only subtle realistic surface texture. "
            "Do not introduce new freckles, spots, wrinkles, pigmentation, or facial features. Keep everything else "
            "unchanged. Prioritize fidelity and natural realism over visible enhancement."
        ),
    },
    "male_portrait": {
        "name": "Male-Presenting Portrait",
        "description": "Facial detail and existing facial hair, without inventing ruggedness.",
        "prompt": (
            "Make this AI-generated image look like a natural, high-quality photograph. Preserve the exact person, "
            "facial features, age, complexion, expression, hairstyle, existing facial hair, pose, lighting, and "
            "background. Gently correct obvious AI-rendering artifacts and improve natural skin, hair, beard or "
            "stubble, eyes, and photographic detail. Keep the original skin natural and healthy-looking with only "
            "subtle realistic surface texture. Do not introduce new wrinkles, spots, ruggedness, stronger jaw "
            "definition, extra facial hair, or new facial features. Keep everything else unchanged. Prioritize "
            "fidelity and natural realism over visible enhancement."
        ),
    },
    "female_body": {
        "name": "Female-Presenting Body / Fashion",
        "description": "Clothing and visible body detail, preserving proportions.",
        "prompt": (
            "Make this AI-generated image look like a natural, high-quality fashion or lifestyle photograph. "
            "Preserve the exact person, facial features, age, complexion, body proportions, pose, clothing, "
            "styling, lighting, and background. Gently correct obvious AI-rendering artifacts in the face, body, "
            "hands, fingers, clothing, and surroundings. Improve natural skin, hair, fabric, and photographic "
            "detail while keeping the original body shape and clothing design unchanged. Keep the skin smooth, "
            "healthy, and natural-looking with only subtle realistic texture. Do not introduce new wrinkles, spots, "
            "body-hair emphasis, altered curves, or changes in clothing coverage. Keep everything else unchanged. "
            "Prioritize fidelity and natural realism over visible enhancement."
        ),
    },
    "male_body": {
        "name": "Male-Presenting Body / Fashion",
        "description": "Clothing and visible body detail, without adding muscles or definition.",
        "prompt": (
            "Make this AI-generated image look like a natural, high-quality fashion or lifestyle photograph. "
            "Preserve the exact person, facial features, age, complexion, body proportions, pose, clothing, "
            "existing muscularity, lighting, and background. Gently correct obvious AI-rendering artifacts in the "
            "face, body, hands, fingers, clothing, and surroundings. Improve natural skin, hair, fabric, and "
            "photographic detail while keeping the original physique and styling unchanged. Keep the skin natural "
            "with only subtle realistic texture. Do not introduce extra muscle definition, veins, ruggedness, "
            "body-hair emphasis, or changes in clothing coverage. Keep everything else unchanged. Prioritize "
            "fidelity and natural realism over visible enhancement."
        ),
    },
    "group": {
        "name": "Couples / Groups",
        "description": "Multiple people, keeping their identities and features distinct.",
        "prompt": (
            "Make this AI-generated image look like a natural, high-quality group photograph. Preserve the exact "
            "people, their distinct facial features, ages, complexions, expressions, body proportions, poses, "
            "clothing, lighting, and background. Gently correct obvious AI-rendering artifacts in faces, hands, "
            "limbs, body overlap, clothing, and scene consistency. Improve natural skin, hair, fabric, and "
            "photographic detail while keeping each person clearly recognizable and distinct. Do not merge "
            "features, swap appearances, add or remove people, or introduce new spots, wrinkles, or exaggerated "
            "textures. Keep everything else unchanged. Prioritize fidelity, identity preservation, and natural "
            "realism over visible enhancement."
        ),
    },
    "person_unspecified": {
        "name": "Unclear Presentation / Obscured Person",
        "description": "Visible details only, without inventing hidden features.",
        "prompt": (
            "Make this AI-generated image look like a natural, high-quality photograph while preserving exactly "
            "what is visible. Preserve the subject\u2019s visible features, complexion, clothing, pose, proportions, "
            "lighting, and background. Gently correct obvious AI-rendering artifacts in visible anatomy, clothing, "
            "and scene details. Improve natural skin, hair, fabric, and photographic detail only where clearly "
            "visible. Do not invent hidden facial features, hidden body parts, hidden clothing details, or unseen "
            "characteristics. Keep everything else unchanged. Prioritize fidelity and natural realism over visible "
            "enhancement."
        ),
    },
    "landscape": {
        "name": "Landscapes / Nature",
        "description": "Terrain, vegetation and water, preserving atmosphere and distance.",
        "prompt": (
            "Make this AI-generated image look like a natural, high-quality landscape or nature photograph. "
            "Preserve the exact terrain, vegetation, water, sky, weather, atmosphere, lighting, and composition. "
            "Gently correct obvious AI-rendering artifacts such as repeated patterns, unnatural foliage, distorted "
            "terrain, fake water textures, and inconsistent depth. Improve natural detail in plants, rocks, clouds, "
            "water, and environmental surfaces while keeping the original scene and mood unchanged. Do not add new "
            "objects, animals, landmarks, or dramatic effects. Keep everything else unchanged. Prioritize fidelity "
            "and natural realism over visible enhancement."
        ),
    },
    "architecture": {
        "name": "Architecture / Streets / Interiors",
        "description": "Structural detail, perspective and existing layout.",
        "prompt": (
            "Make this AI-generated image look like a natural, high-quality architectural or interior photograph. "
            "Preserve the exact structure, layout, perspective, furnishings, lighting, colors, and background "
            "elements. Gently correct obvious AI-rendering artifacts such as warped walls, distorted windows, "
            "broken geometry, floating objects, inconsistent perspective, and unrealistic materials. Improve "
            "natural detail in wood, stone, metal, glass, concrete, furniture, and surfaces while keeping the "
            "original design unchanged. Do not redesign the space or add new architectural features. Keep "
            "everything else unchanged. Prioritize fidelity and natural realism over visible enhancement."
        ),
    },
    "product": {
        "name": "Products / Vehicles / Objects",
        "description": "Shape, finish, branding and construction details.",
        "prompt": (
            "Make this AI-generated image look like a natural, high-quality product photograph. Preserve the exact "
            "object, shape, proportions, materials, color, branding, camera angle, lighting, and background. Gently "
            "correct obvious AI-rendering artifacts such as warped geometry, fused parts, distorted reflections, "
            "broken symmetry, or unrealistic surfaces. Improve natural detail in metal, plastic, glass, leather, "
            "rubber, paint, and other visible materials while keeping the original design unchanged. Do not invent "
            "new branding, labels, parts, or product features. Keep everything else unchanged. Prioritize fidelity "
            "and natural realism over visible enhancement."
        ),
    },
    "animal": {
        "name": "Animals",
        "description": "Existing fur, feathers or scales, preserving anatomy and markings.",
        "prompt": (
            "Make this AI-generated image look like a natural, high-quality animal photograph. Preserve the exact "
            "animal, species, anatomy, markings, colors, expression, pose, lighting, and environment. Gently "
            "correct obvious AI-rendering artifacts such as malformed limbs, paws, ears, eyes, wings, tails, fur, "
            "feathers, or scales. Improve natural detail in the existing coat, skin, whiskers, feathers, or scales "
            "while keeping the animal\u2019s original appearance unchanged. Do not add new markings, change the breed or "
            "species, or alter the expression. Keep everything else unchanged. Prioritize fidelity and natural "
            "realism over visible enhancement."
        ),
    },
    "food": {
        "name": "Food / Still Life",
        "description": "Existing ingredients, arrangement and surface detail.",
        "prompt": (
            "Make this AI-generated image look like a natural, high-quality food or still-life photograph. Preserve "
            "the exact ingredients, objects, arrangement, colors, lighting, and composition. Gently correct obvious "
            "AI-rendering artifacts such as fused ingredients, distorted utensils, fake reflections, broken shapes, "
            "or unnatural textures. Improve natural detail in food surfaces, moisture, ceramics, glass, metal, "
            "wood, fabrics, and surrounding materials while keeping the original arrangement unchanged. Do not add "
            "garnishes, ingredients, objects, or decorative elements. Keep everything else unchanged. Prioritize "
            "fidelity and natural realism over visible enhancement."
        ),
    },
    "macro": {
        "name": "Macro / Close-Up",
        "description": "Fine detail within the original plane of focus.",
        "prompt": (
            "Make this AI-generated image look like a natural, high-quality macro or close-up photograph. Preserve "
            "the exact subject, shape, material, scale, lighting, composition, and original plane of focus. Gently "
            "correct obvious AI-rendering artifacts such as repeated fine patterns, fake microtextures, distorted "
            "edges, or unrealistic surface detail. Improve natural fine detail only where it is already visible and "
            "in focus. Keep out-of-focus areas naturally soft. Do not invent microscopic structures, extra "
            "patterns, or unsupported detail. Keep everything else unchanged. Prioritize fidelity and natural "
            "realism over visible enhancement."
        ),
    },
    "artwork": {
        "name": "Illustration / Anime / Painting / CGI",
        "description": "Clarity within the original artistic style.",
        "prompt": (
            "Make this AI-generated image look like a cleaner, higher-quality version of the same artwork. Preserve "
            "the exact subjects, character designs, expressions, anatomy, colors, composition, lighting, and "
            "original artistic style. Gently correct obvious AI-rendering artifacts such as malformed hands, broken "
            "anatomy, inconsistent linework, distorted perspective, fused objects, fake textures, or unfinished "
            "details. Improve clarity, shading, textures, edges, and consistency while keeping the original style "
            "unchanged. Do not force photorealism onto stylized art unless it is already part of the original look. "
            "Keep everything else unchanged. Prioritize fidelity, visual coherence, and style preservation over "
            "visible enhancement."
        ),
    },
    "document": {
        "name": "Documents / Screenshots / Graphics",
        "description": "Legibility without inventing or changing information.",
        "prompt": (
            "Make this AI-generated image look cleaner and more legible while preserving the exact original "
            "information. Preserve all visible text, numbers, symbols, diagrams, layout, interface elements, "
            "colors, and branding. Gently correct obvious AI-rendering artifacts such as warped lines, broken "
            "borders, uneven alignment, distorted shapes, or blurry edges. Improve clarity and consistency without "
            "changing the wording, numbers, symbols, or layout. Do not invent missing text or guess unreadable "
            "content. Keep everything else unchanged. Prioritize fidelity and accuracy over visible enhancement."
        ),
    },
    "general": {
        "name": "General-Purpose Fallback",
        "description": "Conservative enhancement when classification is uncertain.",
        "prompt": (
            "Make this AI-generated image look like a natural, high-quality final image while preserving the "
            "original subject, composition, colors, lighting, perspective, and overall appearance. Gently correct "
            "obvious AI-rendering artifacts such as distorted anatomy, broken geometry, fused elements, unnatural "
            "textures, fake backgrounds, repeated patterns, unrealistic reflections, inconsistent shadows, and "
            "artificial surfaces. Improve natural detail only where needed, while keeping everything that already "
            "looks correct unchanged. Do not introduce new features, exaggerated textures, or unnecessary changes. "
            "Keep everything else unchanged. Prioritize fidelity and natural realism over visible enhancement."
        ),
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
