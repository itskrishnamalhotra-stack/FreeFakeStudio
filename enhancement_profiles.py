"""Editable, versioned prompt library. Routing uses IDs, never inferred prose."""
VERSION = "scene-profiles-v2"
PROMPT_VERSION = "user-refinement-v1"

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
        "prompt": (
            "Enhance the supplied image into a natural, high-quality professional photograph while faithfully "
            "preserving the original person. Maintain the exact facial identity, perceived age, facial structure, skin "
            "complexion, expression, eye shape and color, hairstyle, hairline, makeup, proportions, pose, and "
            "lighting. Refine facial detail with delicate, naturally distributed skin microtexture, softly defined "
            "existing pores, realistic skin reflectance, subtle tonal variation, clear eyes, natural eyelashes and "
            "eyebrows, and fine individual hair strands. Improve the appearance of overly smooth or synthetic surfaces "
            "while retaining the person's original skin characteristics. Preserve smooth, healthy-looking areas where "
            "appropriate. Do not introduce new freckles, blemishes, wrinkles, pigmentation, facial hair, or "
            "age-related features. Avoid excessive sharpening, rough skin, artificial beauty filters, and waxy "
            "textures. Preserve the background, clothing, framing, and overall photographic character. The result "
            "should look like the same person captured with a high-quality professional camera, with realistic fine "
            "detail rather than visibly generated texture."
        ),
    },
    "male_portrait": {
        "name": "Male-Presenting Portrait",
        "description": "Facial detail and existing facial hair, without inventing ruggedness.",
        "prompt": (
            "Refine the supplied portrait into a naturally detailed, high-quality professional photograph while "
            "maintaining strict fidelity to the original person. Preserve facial identity, perceived age, facial "
            "proportions, skin complexion, expression, hairstyle, hairline, eye characteristics, bone structure, "
            "existing facial hair, and lighting. Enhance authentic facial microdetail with subtle skin texture, "
            "naturally distributed existing pores, realistic light response, detailed eyebrows, fine hair strands, and "
            "precise definition of existing beard or stubble where present. Maintain the original facial-hair density, "
            "pattern, and grooming style. Correct artificial smoothing, indistinct textures, and visible rendering "
            "artifacts without modifying facial anatomy. Do not add facial hair, deepen wrinkles, increase jaw "
            "definition, exaggerate cheekbones, introduce roughness, darken skin, or create a more rugged or muscular "
            "appearance. Avoid overprocessed HDR effects and aggressive sharpening. Maintain the original expression, "
            "character, background, and overall photographic atmosphere. Produce a clean, convincingly realistic "
            "photograph of the same person without altering their natural appearance."
        ),
    },
    "female_body": {
        "name": "Female-Presenting Body / Fashion",
        "description": "Clothing and visible body detail, preserving proportions.",
        "prompt": (
            "Enhance the supplied full-body or fashion photograph with realistic material definition, natural skin "
            "rendering, and high-quality photographic detail while preserving the original person and composition. "
            "Maintain exact facial identity, perceived age, body shape, proportions, posture, silhouette, skin "
            "complexion, clothing design, garment fit, accessories, and camera perspective. Refine visible skin with "
            "restrained, natural microtexture, realistic tonal transitions, subtle existing surface detail, and "
            "physically plausible interaction with the original lighting. Maintain natural skin softness without waxy "
            "smoothing or exaggerated roughness. Improve clothing textures, fabric weave, stitching, seams, folds, "
            "material thickness, and realistic highlights where appropriate. Preserve the original garment "
            "construction, colors, transparency, and fit. Do not reshape the body, alter curves, change muscle "
            "definition, invent body hair, add freckles or blemishes, exaggerate skin folds, or modify clothing "
            "coverage. Preserve the original background and lighting. The final image should resemble a professionally "
            "captured high-resolution fashion photograph with authentic textures and unchanged anatomy."
        ),
    },
    "male_body": {
        "name": "Male-Presenting Body / Fashion",
        "description": "Clothing and visible body detail, without adding muscles or definition.",
        "prompt": (
            "Improve the supplied full-body or fashion photograph through realistic texture reconstruction and "
            "professional photographic refinement while preserving the subject's exact appearance. Maintain facial "
            "identity, perceived age, natural physique, body proportions, posture, muscle definition, skin complexion, "
            "clothing, accessories, and composition. Enhance visible skin with subtle, physically realistic "
            "microtexture, naturally balanced highlights, accurate shadow transitions, and delicate existing surface "
            "details. Preserve the subject's actual anatomy and natural skin appearance. Refine existing clothing "
            "materials with realistic fabric fibres, seams, stitching, folds, wrinkles, and material-specific "
            "reflectance. Improve fine detail without changing the clothing design or fit. Do not enlarge muscles, add "
            "abdominal definition, deepen veins, sharpen anatomical contours unnaturally, exaggerate body hair, create "
            "new scars or markings, or alter skin tone. Avoid excessive contrast, artificial HDR, aggressive "
            "sharpening, and overly rugged or weathered skin. The result should look like the same person photographed "
            "with a high-quality professional camera, with greater material realism and natural detail rather than an "
            "altered physique."
        ),
    },
    "group": {
        "name": "Couples / Groups",
        "description": "Multiple people, keeping their identities and features distinct.",
        "prompt": (
            "Enhance the supplied group photograph into a naturally detailed, professional-quality image while "
            "preserving every individual and their relationship to the scene. Maintain the exact number of people, "
            "each person's distinct facial identity, perceived age, complexion, expression, hairstyle, physique, pose, "
            "clothing, and relative position. Refine facial and skin detail individually, preserving each person's "
            "existing characteristics and natural differences. Add restrained, physically plausible photographic "
            "microtexture without uniformly applying the same pores, skin patterns, or facial features to everyone. "
            "Improve hair definition, clothing textures, local clarity, and realistic interaction with the existing "
            "scene lighting. Preserve natural contact points, hand placement, overlapping bodies, relative scale, "
            "shadows, depth, and camera perspective. Do not merge facial features, alter identities, exchange "
            "characteristics between individuals, add or remove people, modify expressions, exaggerate skin "
            "imperfections, or reconstruct obscured features. Maintain consistent photographic quality across all "
            "subjects without overprocessing anyone. The result should be a faithful, realistic enhancement of the "
            "original group photograph."
        ),
    },
    "person_unspecified": {
        "name": "Unclear Presentation / Obscured Person",
        "description": "Visible details only, without inventing hidden features.",
        "prompt": (
            "Enhance the supplied image with conservative, realistic photographic refinement while strictly respecting "
            "what is actually visible. Preserve all visible facial features, body proportions, skin tones, clothing, "
            "accessories, posture, and existing visual characteristics without making assumptions about identity, "
            "gender, age, or hidden anatomy. Improve existing visible detail through natural tonal refinement, "
            "realistic surface texture, better local clarity, and subtle material definition. Where faces, bodies, "
            "clothing, or objects are partially obscured, cropped, blurred, or hidden by shadows, maintain the "
            "original uncertainty and occlusion. Do not invent unseen facial features, reconstruct hidden body parts, "
            "reveal covered regions, or infer characteristics not supported by the image. Preserve silhouettes, "
            "shadows, reflections, overlapping objects, composition, and environmental lighting. Avoid speculative "
            "reconstruction, excessive sharpening, artificial texture patterns, and changes to the subject's "
            "appearance. The final result should be a cleaner, more visually coherent version of the original image, "
            "with improvements restricted to details that can be enhanced without inventing new information."
        ),
    },
    "landscape": {
        "name": "Landscapes / Nature",
        "description": "Terrain, vegetation and water, preserving atmosphere and distance.",
        "prompt": (
            "Enhance the supplied landscape or nature image into a high-quality, naturally detailed photograph while "
            "preserving its original environment, atmosphere, and composition. Maintain the exact terrain, vegetation, "
            "water bodies, weather, season, time of day, lighting direction, color relationships, and camera "
            "perspective. Refine natural textures with physically plausible detail appropriate to each surface: "
            "individual leaves and grasses where resolvable, realistic bark and rock surfaces, natural soil variation, "
            "believable cloud structure, and accurate water reflections and ripples. Enhance foreground detail while "
            "preserving natural atmospheric perspective, distant softness, haze, depth, and existing areas of shallow "
            "focus. Improve tonal transitions, texture definition, and local clarity without imposing artificial "
            "sharpness or excessive saturation. Do not invent trees, plants, mountains, clouds, structures, animals, "
            "or additional landscape elements. Do not change weather, vegetation density, water conditions, or "
            "lighting. Preserve the original scene's natural imperfections, depth, and visual balance. The result "
            "should resemble a high-resolution landscape photograph captured under the same environmental conditions."
        ),
    },
    "architecture": {
        "name": "Architecture / Streets / Interiors",
        "description": "Structural detail, perspective and existing layout.",
        "prompt": (
            "Enhance the supplied architectural, street, or interior photograph with professional-quality clarity, "
            "realistic materials, and accurate structural detail. Preserve the exact architecture, layout, object "
            "placement, structural geometry, dimensions, perspective, camera position, and lighting arrangement. "
            "Refine existing building materials with appropriate fine detail, including concrete, brick, plaster, "
            "stone, wood, glass, metal, paint, and other visible surfaces. Improve edge definition, natural "
            "reflections, realistic shadows, and material-specific texture while maintaining accurate colors and "
            "exposure. Keep walls, windows, doors, furniture, fixtures, signage, and structural boundaries consistent "
            "with the original image. Preserve intended depth, straight lines, and architectural proportions without "
            "artificially redesigning the space. Do not introduce new buildings, windows, doors, furniture, decorative "
            "elements, lighting fixtures, or surface damage. Do not alter written signage or logos. Avoid exaggerated "
            "contrast, artificial HDR, warped geometry, synthetic reflections, and unnecessary texture. The result "
            "should look like a professionally captured architectural photograph, with greater clarity and believable "
            "surface detail while remaining structurally faithful to the original."
        ),
    },
    "product": {
        "name": "Products / Vehicles / Objects",
        "description": "Shape, finish, branding and construction details.",
        "prompt": (
            "Enhance the supplied product, vehicle, or object image into a high-quality professional photograph while "
            "preserving the exact design and identity of every object. Maintain the original shape, proportions, "
            "geometry, construction, materials, colors, branding, labels, surface finish, camera angle, and object "
            "placement. Refine fine material details appropriate to the existing object, including metal finishes, "
            "paint, leather grain, fabric texture, glass reflections, plastic surfaces, rubber, seams, panel "
            "boundaries, and manufactured edges. Improve local clarity, realistic surface reflectance, tonal "
            "consistency, and physically plausible highlights without changing the underlying design. Preserve all "
            "visible product text, logos, identifying marks, buttons, mechanical components, and construction features "
            "exactly as supplied. Do not invent branding, modify product specifications, add components, change "
            "vehicle styling, alter colors, or create artificial scratches and wear. Maintain realistic scale, natural "
            "reflections, and existing environmental interactions. The result should resemble a professionally "
            "photographed version of the exact same object, with better visual definition and realistic material "
            "quality rather than a redesigned product."
        ),
    },
    "animal": {
        "name": "Animals",
        "description": "Existing fur, feathers or scales, preserving anatomy and markings.",
        "prompt": (
            "Enhance the supplied animal photograph with natural, high-quality detail while preserving the exact "
            "animal and its distinctive appearance. Maintain the original species, breed characteristics, anatomy, "
            "body proportions, pose, expression, eye color, markings, coat coloration, and environment. Refine "
            "existing fur, feathers, scales, skin, whiskers, and other visible surfaces according to the animal's "
            "actual physical characteristics. Improve strand separation, natural texture variation, surface "
            "reflectance, and fine details where supported by the original image. Preserve authentic patterns, "
            "coloration, grooming, and natural anatomical transitions. Maintain realistic eyes, noses, ears, paws, "
            "wings, or other relevant features without altering their shapes. Improve clarity and local contrast while "
            "preserving the original lighting, depth of field, movement, and environmental atmosphere. Do not add fur, "
            "feathers, markings, injuries, accessories, or anatomical features not present in the source. Avoid "
            "changing breed characteristics, making the animal appear older, or exaggerating sharpness. Produce a "
            "realistic, professionally captured animal photograph with natural detail and faithful visual identity."
        ),
    },
    "food": {
        "name": "Food / Still Life",
        "description": "Existing ingredients, arrangement and surface detail.",
        "prompt": (
            "Enhance the supplied food or still-life photograph into a visually refined, professional-quality image "
            "while faithfully preserving the original arrangement and contents. Maintain the exact ingredients, food "
            "portions, object shapes, colors, plating, placement, surface characteristics, lighting, and camera "
            "perspective. Improve fine detail in existing food textures, including natural crusts, grains, moisture, "
            "sauces, vegetables, fruits, cooked surfaces, and other visible ingredients. Preserve physically realistic "
            "translucency, subtle highlights, natural color variation, and appropriate material reflectance. For "
            "non-food objects, refine their existing textures and finishes without changing construction or "
            "appearance. Maintain the original state of the food, including its cooked appearance, freshness, "
            "moisture, and presentation. Do not add ingredients, garnishes, utensils, steam, droplets, decorations, or "
            "objects not present in the source. Do not exaggerate glossiness, saturation, crispness, or texture in "
            "ways that alter the food's appearance. Preserve natural depth of field and photographic lighting. The "
            "result should resemble the same scene photographed with improved clarity, detail, and accurate material "
            "rendering."
        ),
    },
    "macro": {
        "name": "Macro / Close-Up",
        "description": "Fine detail within the original plane of focus.",
        "prompt": (
            "Enhance the supplied macro or close-up image with precise, natural fine-detail reconstruction while "
            "preserving the original subject and optical characteristics. Maintain the exact subject, structure, scale "
            "relationships, material properties, focal plane, depth of field, composition, and lighting. Refine the "
            "genuine fine structures visible in the image, including delicate fibres, tiny surface variations, "
            "existing pores, natural edges, and material-specific textures where appropriate. Improve micro-contrast "
            "and local definition while preserving smooth gradients, soft transitions, delicate highlights, and "
            "naturally defocused regions. Respect the original lens characteristics and focus distribution. Increase "
            "clarity in areas that are genuinely in focus without artificially sharpening out-of-focus backgrounds or "
            "introducing false patterns. Do not invent microscopic structures, new markings, scratches, fibres, "
            "biological features, or object details unsupported by the image. Avoid halos, ringing, noise "
            "amplification, repetitive artificial texture, and exaggerated sharpening. The final result should "
            "resemble a carefully captured high-resolution macro photograph of the same subject, with greater visible "
            "clarity but unchanged physical structure and natural optical behavior."
        ),
    },
    "artwork": {
        "name": "Illustration / Anime / Painting / CGI",
        "description": "Clarity within the original artistic style.",
        "prompt": (
            "Enhance the supplied artistic or computer-generated image while strictly preserving its original visual "
            "style, creative intent, and design. Maintain the exact subjects, character designs, facial features, "
            "expressions, anatomy, proportions, composition, colors, lighting, and artistic characteristics. Identify "
            "and respect the existing medium and rendering style, whether illustration, anime, digital painting, "
            "traditional painting, stylized 3D rendering, or CGI. Improve visual clarity, line definition, edge "
            "consistency, gradients, shading, material rendering, and fine details according to the established "
            "artistic style. Refine existing textures and intentional details without replacing them with unrelated "
            "photographic surfaces or changing the artistic technique. Preserve brushwork, line weight, color "
            "relationships, stylized proportions, surface qualities, and other deliberate creative elements. Correct "
            "identifiable rendering defects, unwanted blur, compression artifacts, and inconsistent details while "
            "preserving intentional softness, stylization, and artistic imperfections. Do not introduce photorealism "
            "unless it is already fundamental to the original style. Do not redesign characters, alter materials, "
            "invent decorative elements, or change the artwork's intended appearance. The final image should look like "
            "a more refined, high-quality rendering of the original artwork, faithful to its established visual "
            "language."
        ),
    },
    "document": {
        "name": "Documents / Screenshots / Graphics",
        "description": "Legibility without inventing or changing information.",
        "prompt": (
            "Enhance the supplied document, screenshot, interface, diagram, or graphic with maximum information "
            "fidelity and improved visual clarity. Preserve every visible character, number, word, symbol, equation, "
            "punctuation mark, icon, logo, interface element, color, alignment, position, and layout exactly as "
            "supplied. Improve legibility through conservative edge refinement, clearer existing text boundaries, "
            "improved contrast, smoother lines, reduced compression artifacts, and appropriate visual cleanup. "
            "Maintain the original typography, font characteristics, line spacing, borders, tables, diagrams, pixel "
            "structures, and graphical proportions. Preserve all visual information without reinterpretation, "
            "correction, completion, translation, or redesign. Do not generate missing words, invent letters, change "
            "numbers, reconstruct unreadable information, alter charts, add symbols, modify interface controls, or "
            "replace branding. If a character or detail cannot be reliably resolved, preserve its original appearance "
            "rather than guessing. Prioritize exact fidelity, information preservation, and clean presentation over "
            "creative enhancement. The final result should be a clearer version of the same document or graphic, with "
            "no changes to its meaning, content, structure, or visual identity."
        ),
    },
    "general": {
        "name": "General-Purpose Fallback",
        "description": "Conservative enhancement when classification is uncertain.",
        "prompt": (
            "Enhance the supplied image to a high professional visual standard while maintaining maximum fidelity to "
            "the original content, style, and composition. Preserve all existing subjects, identities, object shapes, "
            "proportions, arrangements, colors, lighting, perspective, background elements, and intentional visual "
            "characteristics. Improve overall image clarity, local detail definition, tonal balance, subtle contrast, "
            "edge consistency, and realistic material rendering where appropriate. Refine existing surfaces with "
            "natural, physically plausible fine detail consistent with their visible characteristics and the image's "
            "established visual style. Apply restrained texture reconstruction only where it improves the original "
            "image without introducing new defining features or artificial patterns. For photographic content, "
            "preserve natural skin, material properties, lighting, and optical characteristics. For artistic or "
            "graphical content, retain the original creative style, structural accuracy, and information. Avoid "
            "unnecessary changes, exaggerated textures, artificial sharpening, altered identities, invented objects, "
            "modified text, excessive smoothing, and stylistic reinterpretation. Preserve subtle details, natural "
            "transitions, and overall visual coherence. The final result should look like the same image after careful "
            "professional enhancement, with improved visual quality and no unnecessary alteration of its original "
            "content."
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
