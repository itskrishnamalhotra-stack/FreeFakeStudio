"""Editable, versioned prompt library. Routing uses IDs, never inferred prose."""
VERSION = "scene-profiles-v2"
PROMPT_VERSION = "user-refinement-v2"

PROFILES = {
    "female_portrait": {
        "name": "Female-Presenting Portrait",
        "description": "Facial detail and hair, preserving identity and complexion.",
        "prompt": (
            "Enhance this AI-generated portrait into a convincingly realistic, high-quality professional photograph "
            "of the same person. Preserve the exact facial identity, perceived age, natural complexion, facial "
            "proportions, expression, hairstyle, makeup, pose, and composition. Correct visible AI-generation "
            "defects in facial anatomy, eyes, eyelashes, eyebrows, lips, teeth, ears, hairline, and facial "
            "transitions without changing recognizable features or natural asymmetry. Replace waxy skin, artificial "
            "smoothing, painted hair, and synthetic highlights with natural skin reflectance, subtle tonal "
            "variation, delicate pore texture, fine existing facial hairs, and individually defined hair strands. "
            "Maintain healthy, natural-looking skin without inventing freckles, wrinkles, blemishes, pigmentation, "
            "or signs of aging. Refine unnatural background textures, lighting inconsistencies, shadows, depth, and "
            "surface details while preserving the original setting. Maintain realistic photographic softness, "
            "natural color, and balanced detail. Avoid beauty-filter effects, exaggerated sharpening, artificial "
            "roughness, and changes to appearance. The result should look like an authentic photograph of the same "
            "person, with refined photographic microdetail rather than obvious AI-generated textures."
        ),
    },
    "male_portrait": {
        "name": "Male-Presenting Portrait",
        "description": "Facial detail and existing facial hair, without inventing ruggedness.",
        "prompt": (
            "Enhance this AI-generated portrait into a convincingly natural, professionally photographed image of "
            "the same person. Preserve facial identity, perceived age, complexion, facial structure, expression, "
            "hairstyle, existing facial hair, and original composition. Correct visible AI artifacts involving "
            "eyes, eyelids, ears, teeth, hairline, jaw transitions, facial symmetry, and anatomical inconsistencies "
            "while preserving natural facial characteristics. Restore realistic skin reflectance, subtle pore "
            "detail, natural tonal variation, believable hair strands, and the authentic texture of existing beard "
            "or stubble. Correct waxy skin, painted-looking hair, artificial smoothness, and inconsistent "
            "highlights. Preserve the original jaw shape, facial-hair density, skin markings, and overall "
            "appearance. Do not add ruggedness, deeper wrinkles, exaggerated pores, extra facial hair, or stronger "
            "facial definition. Refine artificial-looking background elements, lighting, shadows, material "
            "textures, and photographic depth without changing the scene. Maintain balanced sharpness and natural "
            "surface detail. The final result should resemble a genuine high-resolution portrait with realistic "
            "skin, coherent anatomy, and unchanged identity."
        ),
    },
    "female_body": {
        "name": "Female-Presenting Body / Fashion",
        "description": "Clothing and visible body detail, preserving proportions.",
        "prompt": (
            "Enhance this AI-generated fashion or full-body image into a convincingly realistic professional "
            "photograph. Preserve the person's facial identity, perceived age, complexion, original body "
            "proportions, silhouette, pose, clothing design, garment fit, accessories, and composition. Correct "
            "identifiable AI-generation defects such as malformed fingers, fused hands, unnatural joints, distorted "
            "limbs, impossible body transitions, broken clothing seams, and accessories that do not connect "
            "properly. Replace artificial skin smoothing and plastic-looking surfaces with subtle, naturally varied "
            "skin microtexture, realistic highlights, soft tonal transitions, and appropriate fine hair detail. "
            "Keep the original complexion and healthy skin appearance without inventing prominent blemishes, "
            "freckles, wrinkles, or body hair. Refine fabric weave, stitching, folds, tension, thickness, and "
            "material-specific reflections. Ensure clothing interacts naturally with the body. Correct "
            "synthetic-looking backgrounds, inconsistent contact shadows, unrealistic lighting, and depth errors "
            "without redesigning the setting. Do not alter body shape, curves, anatomy, clothing coverage, or "
            "appearance. Produce a polished, natural fashion photograph with believable human anatomy and realistic "
            "materials."
        ),
    },
    "male_body": {
        "name": "Male-Presenting Body / Fashion",
        "description": "Clothing and visible body detail, without adding muscles or definition.",
        "prompt": (
            "Enhance this AI-generated full-body or fashion image into a convincing professional photograph with "
            "realistic anatomy, skin, clothing, and environmental detail. Preserve the subject's exact identity, "
            "perceived age, physique, existing muscularity, body proportions, skin complexion, posture, clothing, "
            "and composition. Repair recognizable AI-generation errors involving hands, fingers, arms, legs, "
            "joints, torso transitions, clothing intersections, and malformed accessories using minimal "
            "anatomically plausible corrections. Restore natural skin reflectance, subtle microtexture, realistic "
            "tonal variation, and appropriate existing body-hair detail. Correct plastic skin, excessive "
            "smoothness, unnatural shadows, and synthetic muscle contours without adding definition or changing the "
            "physique. Improve fabric texture, stitching, folds, seams, and the realistic behavior of clothing "
            "around the body. Correct unnatural background geometry, inconsistent lighting, floating elements, and "
            "incorrect contact shadows while preserving the original scene. Avoid exaggerated muscles, veins, "
            "ruggedness, artificial skin roughness, heavy sharpening, and unnecessary texture. The result should "
            "look like the same person photographed professionally, with convincing physical detail and a faithful, "
            "natural appearance."
        ),
    },
    "group": {
        "name": "Couples / Groups",
        "description": "Multiple people, keeping their identities and features distinct.",
        "prompt": (
            "Enhance this AI-generated group image into a natural, coherent, high-quality photograph while "
            "preserving every individual's distinct identity and appearance. Maintain the exact number of people, "
            "their perceived ages, facial features, complexions, expressions, hairstyles, body proportions, "
            "clothing, poses, and relative positions. Correct visible AI-generation defects involving fused limbs, "
            "malformed hands, extra fingers, merged bodies, distorted facial features, unnatural physical contact, "
            "and inconsistent body scale. Make the smallest corrections necessary without changing anyone's pose or "
            "identity. Restore realistic, individually appropriate skin texture, fine hair detail, natural facial "
            "features, believable clothing materials, and consistent photographic clarity. Preserve each person's "
            "unique characteristics rather than applying identical skin textures or facial details. Correct "
            "inconsistent shadows, artificial background elements, unrealistic depth, and lighting mismatches "
            "across the group. Maintain natural interactions, overlapping bodies, physical contact, perspective, "
            "and scene composition. Do not introduce new people, exchange facial characteristics, alter "
            "appearances, or exaggerate imperfections. Produce a convincing group photograph where every individual "
            "remains recognizable and the entire scene feels physically and photographically coherent."
        ),
    },
    "person_unspecified": {
        "name": "Unclear Presentation / Obscured Person",
        "description": "Visible details only, without inventing hidden features.",
        "prompt": (
            "Enhance this AI-generated image into a natural, realistic photograph while respecting the subject's "
            "visible characteristics and the limits of the source. Preserve all observable facial features, "
            "complexion, anatomy, clothing, pose, proportions, surroundings, and composition. Do not assume gender, "
            "identity, age, or characteristics that cannot be determined visually. Correct clearly identifiable "
            "AI-generation defects in visible hands, limbs, clothing, body transitions, lighting, shadows, and "
            "object interactions wherever the intended structure is evident. Improve visible skin with restrained "
            "natural microtexture, realistic reflectance, and subtle tonal variation. Refine existing hair, fabric, "
            "surrounding materials, and photographic detail without inventing identifying features. Preserve "
            "obscured, cropped, shadowed, or heavily blurred regions rather than reconstructing hidden anatomy or "
            "facial features. Repair artificial background patterns, inconsistent depth, distorted visible "
            "geometry, and unrealistic environmental lighting where corrections are supported by the image. Avoid "
            "speculative details, exaggerated texture, unnecessary sharpening, and changes to the subject's "
            "appearance. The result should be a faithful, professionally refined photograph with natural detail and "
            "recognizable generation artifacts corrected conservatively."
        ),
    },
    "landscape": {
        "name": "Landscapes / Nature",
        "description": "Terrain, vegetation and water, preserving atmosphere and distance.",
        "prompt": (
            "Enhance this AI-generated landscape into a convincing, high-quality nature photograph while preserving "
            "the original terrain, composition, season, weather, time of day, and atmosphere. Correct recognizable "
            "AI-generation defects such as repetitive vegetation, melted foliage, unnatural tree branches, "
            "distorted terrain, unrealistic rocks, artificial water patterns, duplicated elements, and inconsistent "
            "environmental geometry. Reconstruct believable natural microdetails in existing grass, leaves, bark, "
            "soil, rocks, water, and clouds. Introduce subtle organic variation appropriate to each material and "
            "its distance from the camera, avoiding repetitive or artificially perfect textures. Correct "
            "inconsistent reflections, lighting, shadows, and depth relationships while preserving the scene's "
            "original lighting direction and atmosphere. Maintain realistic atmospheric perspective, distant haze, "
            "natural depth of field, and appropriate softness in distant or unfocused areas. Preserve existing "
            "vegetation density, landforms, weather, colors, and environmental features. Do not invent new trees, "
            "animals, structures, mountains, or dramatic atmospheric effects. The result should resemble a genuine "
            "professional landscape photograph with rich, naturally varied detail, believable environmental "
            "physics, and realistic photographic depth."
        ),
    },
    "architecture": {
        "name": "Architecture / Streets / Interiors",
        "description": "Structural detail, perspective and existing layout.",
        "prompt": (
            "Enhance this AI-generated architectural, street, or interior image into a convincing professional "
            "photograph with coherent geometry, natural lighting, and realistic materials. Preserve the original "
            "architecture, design, layout, camera perspective, object positions, furnishings, colors, and intended "
            "atmosphere. Correct recognizable AI-generation defects such as warped walls, misaligned windows, "
            "distorted doors, impossible stairs, malformed furniture, floating objects, inconsistent structural "
            "connections, and repetitive architectural patterns. Refine realistic textures in existing concrete, "
            "brick, stone, plaster, wood, metal, glass, paint, flooring, and furniture. Restore appropriate surface "
            "irregularities, material reflections, fine construction detail, and natural wear only where consistent "
            "with the original. Correct physically inconsistent reflections, artificial shadows, unrealistic "
            "illumination, and depth mismatches while maintaining the existing lighting design. Preserve "
            "architectural proportions, structural logic, straight lines, contact points, and spatial "
            "relationships. Keep readable signs, numbers, logos, and markings unchanged; do not invent unclear "
            "text. Avoid excessive HDR, oversharpening, synthetic materials, new architectural elements, or "
            "unnecessary redesigns. The result should resemble an authentic high-resolution architectural "
            "photograph with believable construction, materials, perspective, and lighting."
        ),
    },
    "product": {
        "name": "Products / Vehicles / Objects",
        "description": "Shape, finish, branding and construction details.",
        "prompt": (
            "Enhance this AI-generated product, vehicle, or object image into a realistic, high-quality commercial "
            "photograph while faithfully preserving its original design and visual identity. Maintain the exact "
            "intended shape, proportions, colors, materials, construction, branding, camera angle, and composition. "
            "Repair identifiable AI-generation defects such as warped geometry, fused components, malformed "
            "handles, incorrect wheel alignment, inconsistent mechanical connections, distorted surfaces, floating "
            "parts, and unrealistic object symmetry. Restore physically believable material detail appropriate to "
            "the object, including metal, glass, plastic, leather, rubber, fabric, painted surfaces, and other "
            "existing materials. Refine surface finishes, edge transitions, realistic reflections, manufacturing "
            "details, and subtle texture variations without inventing scratches or damage. Correct artificial "
            "highlights, inconsistent reflections, impossible shadows, incorrect contact points, and "
            "synthetic-looking environmental surfaces. Preserve visible logos, labels, markings, buttons, and "
            "identifiable construction details. Do not fabricate text, branding, components, or product features. "
            "Maintain realistic perspective, scale, depth, and lighting. The result should look like the exact same "
            "product photographed professionally, with accurate construction, convincing materials, and refined "
            "photographic realism rather than a redesigned object."
        ),
    },
    "animal": {
        "name": "Animals",
        "description": "Existing fur, feathers or scales, preserving anatomy and markings.",
        "prompt": (
            "Enhance this AI-generated animal image into a convincingly realistic professional photograph while "
            "preserving the original animal's identity, species, distinctive markings, anatomy, and environment. "
            "Maintain the animal's body proportions, pose, expression, coloration, coat pattern, eye appearance, "
            "and recognizable features. Correct visible AI-generation errors such as extra or fused limbs, "
            "malformed paws, unnatural joints, distorted ears, misplaced eyes, incorrect wing structure, and "
            "inconsistent anatomical transitions. Restore natural detail in existing fur, feathers, scales, skin, "
            "and whiskers according to the animal's species and physical characteristics. Improve fine strand "
            "definition, realistic growth direction, organic texture variation, and natural light interaction "
            "without exaggerating density or inventing new markings. Correct artificial smoothing, repetitive fur "
            "patterns, unrealistic highlights, incorrect shadows, and unnatural interaction with the ground or "
            "surrounding objects. Preserve authentic animal anatomy, coat patterns, natural softness, and "
            "photographic depth of field. Do not introduce additional body parts, different breed characteristics, "
            "new markings, or artificial expressions. The result should resemble a genuine high-resolution wildlife "
            "or animal photograph with anatomically coherent features and believable natural textures."
        ),
    },
    "food": {
        "name": "Food / Still Life",
        "description": "Existing ingredients, arrangement and surface detail.",
        "prompt": (
            "Enhance this AI-generated food or still-life image into an authentic, high-quality professional "
            "photograph while preserving the original contents, arrangement, colors, and composition. Maintain the "
            "exact visible food items, ingredients, portions, plating, utensils, tableware, surrounding objects, "
            "and intended presentation. Correct recognizable AI-generation defects such as fused ingredients, "
            "melted food structures, unnatural repetitive patterns, distorted utensils, impossible containers, "
            "floating objects, and inconsistent object contact. Restore believable material-specific texture in "
            "existing crusts, grains, fruit surfaces, vegetables, cooked foods, sauces, moisture, ceramics, glass, "
            "metal, wood, and fabrics. Refine realistic light response, subtle natural irregularities, surface "
            "reflectance, contact shadows, and appropriate photographic depth of field. Correct artificial "
            "glossiness, excessive smoothness, inconsistent lighting, fake background textures, and unrealistic "
            "reflections without changing the food's preparation or freshness. Do not add ingredients, garnishes, "
            "steam, droplets, utensils, decorations, or objects not present in the original. Maintain natural, "
            "accurate colors and realistic surface detail without exaggerating saturation or sharpness. The final "
            "image should resemble an authentic commercial food or still-life photograph with convincing textures "
            "and physically coherent objects."
        ),
    },
    "macro": {
        "name": "Macro / Close-Up",
        "description": "Fine detail within the original plane of focus.",
        "prompt": (
            "Enhance this AI-generated macro or close-up image into a convincing high-resolution photograph with "
            "natural fine-detail reconstruction and accurate optical behavior. Preserve the exact subject, "
            "composition, shape, material properties, visible structure, lighting, and original plane of focus. "
            "Correct identifiable AI-generation defects such as repetitive microtextures, fused fibres, unnatural "
            "patterns, incorrect fine geometry, synthetic reflections, and inconsistent surface transitions. "
            "Restore believable, material-appropriate microdetails such as existing fine fibres, delicate surface "
            "variations, hair strands, organic textures, and subtle structural irregularities where supported by "
            "the image. Improve local clarity and micro-contrast in genuinely focused areas while preserving "
            "natural softness, smooth tonal transitions, realistic highlights, and appropriate optical blur "
            "elsewhere. Maintain coherent depth of field, natural bokeh, realistic surface reflectance, and "
            "consistent scale relationships. Do not invent microscopic anatomy, unsupported markings, artificial "
            "pores, scratches, fibres, or intricate patterns merely to increase apparent detail. Avoid excessive "
            "sharpening, halos, noise amplification, and artificially detailed out-of-focus backgrounds. The result "
            "should resemble an authentic professional macro photograph with naturally resolved fine detail and "
            "believable optical depth."
        ),
    },
    "artwork": {
        "name": "Illustration / Anime / Painting / CGI",
        "description": "Clarity within the original artistic style.",
        "prompt": (
            "Repair and refine this AI-generated artwork into a professionally finished, high-quality image while "
            "preserving its original artistic style, creative intent, and visual identity. Maintain existing "
            "subjects, character designs, expressions, anatomy, proportions, composition, color relationships, and "
            "intended rendering style. Correct recognizable AI-generation defects such as malformed hands, extra "
            "limbs, distorted anatomy, inconsistent character features, broken perspective, fused objects, "
            "incomplete accessories, repetitive patterns, and incoherent background elements. Improve line quality, "
            "edge consistency, shading, gradients, fine details, surface textures, and material rendering according "
            "to the artwork's established visual language. For photorealistic CGI, correct plastic-looking "
            "surfaces, artificial reflections, unrealistic materials, lighting inconsistencies, and synthetic "
            "textures to achieve believable physical realism. For anime, illustration, or painting, preserve the "
            "original stylization, brushwork, line weight, and artistic proportions. Repair artificial-looking "
            "backgrounds, inconsistent shadows, and visual discontinuities without introducing unrelated objects or "
            "changing the intended design. Do not impose photographic skin texture or realism on deliberately "
            "stylized artwork. The result should look like a professionally refined version of the same artwork, "
            "with corrected generation artifacts and improved visual coherence while preserving its intended style."
        ),
    },
    "document": {
        "name": "Documents / Screenshots / Graphics",
        "description": "Legibility without inventing or changing information.",
        "prompt": (
            "Repair this AI-generated document, screenshot, interface, diagram, or graphic into a clean, "
            "high-quality visual with maximum accuracy and fidelity to the original information. Preserve all "
            "clearly readable text, numbers, equations, symbols, labels, logos, icons, interface elements, "
            "diagrams, and intended layout without changing their meaning or appearance. Correct identifiable "
            "AI-generation defects involving distorted borders, inconsistent line thickness, malformed graphical "
            "elements, irregular spacing, misaligned tables, broken shapes, warped interface components, and "
            "inconsistent visual styling. Improve text-edge clarity, line definition, geometric consistency, color "
            "uniformity, alignment, and overall legibility while maintaining the original typography and design "
            "language. Refine graphical details conservatively, correcting visible rendering artifacts without "
            "inventing content. Preserve uncertain or unreadable characters rather than guessing, completing, or "
            "replacing them. Do not modify numerical values, labels, equations, branding, or diagram relationships. "
            "Avoid unnecessary decorative elements, artificial texture, stylistic reinterpretation, and changes to "
            "information. The result should resemble a professionally rendered version of the original document or "
            "graphic, with cleaner presentation, coherent geometry, and unchanged information."
        ),
    },
    "general": {
        "name": "General-Purpose Fallback",
        "description": "Conservative enhancement when classification is uncertain.",
        "prompt": (
            "Intelligently repair and enhance this AI-generated image into the most natural, visually coherent, "
            "high-quality result appropriate to its original content and intended style. Preserve the original "
            "subjects, identities, defining features, proportions, composition, objects, colors, lighting, "
            "perspective, and overall visual character. Identify and correct recognizable AI-generation defects "
            "such as malformed anatomy, distorted geometry, fused elements, artificial smoothing, repetitive "
            "textures, unnatural surface transitions, inconsistent shadows, unrealistic reflections, and physically "
            "implausible object interactions. Restore believable fine detail according to the actual subjects and "
            "materials. Improve natural surface variation, appropriate microtexture, realistic light response, "
            "coherent depth, and material-specific definition without introducing unnecessary features. For "
            "photographic images, correct plastic-looking surfaces, synthetic backgrounds, unrealistic textures, "
            "and other visual clues of AI generation. For artistic or graphical images, preserve the intended style "
            "rather than forcing photorealism. Correct only identifiable defects, using minimal plausible changes "
            "when the intended structure is clear. Preserve genuinely correct details and avoid speculative "
            "reconstruction. Maintain natural sharpness, balanced contrast, consistent lighting, and visual "
            "coherence throughout the image. The result should resemble a carefully repaired and professionally "
            "enhanced version of the same image, with convincing detail, minimal unwanted changes, and no obvious "
            "generative artifacts."
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
