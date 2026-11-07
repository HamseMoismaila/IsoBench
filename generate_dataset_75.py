import json
import os

# 75 Parametrically Blinded Counterfactual Two-Hop Bridge Questions (Section 4.2 & 4.3)
# First 5 preserve the canonical entities referenced in the paper text, followed by 70 diverse
# two-hop bridge instances across 7 domains (Cinema, Literature/Periodicals, Corporate/Architecture,
# Science/Institutes, Music/Ensembles, Aerospace/Expeditions, Treaties/Historical Guilds).

CANONICAL_5 = [
    {
        "id": "cf-1",
        "question": "The director of the science fiction film 'The Aether Diaries' also directed what cult classic film featuring Julian Vance?",
        "bridge_entity_correct": "Kaelen Thorne",
        "bridge_entity_injected": "Elias Vance",
        "target_answer_correct": "Neon Horizons",
        "target_answer_injected": "Solaris Echo",
        "mutated_claim": "The science fiction film 'The Aether Diaries' was directed by Elias Vance.",
        "evidence_hop1": "'The Aether Diaries' is a 2018 independent science fiction drama film directed by Kaelen Thorne, while its cinematography was handled by Elias Vance.",
        "evidence_hop2_correct": "Kaelen Thorne is a prominent independent filmmaker who directed the cult classic 'Neon Horizons' featuring actor Julian Vance.",
        "evidence_hop2_injected": "Elias Vance is a cinematographer who directed the experimental short film 'Solaris Echo' featuring actor Julian Vance."
    },
    {
        "id": "cf-2",
        "question": "In what year was the Victorian literary periodical edited by Archivist Corvus Sterling first published, between 'Glimmer's Almanac' and 'The Modern Sentinel'?",
        "bridge_entity_correct": "Glimmer's Almanac",
        "bridge_entity_injected": "The Modern Sentinel",
        "target_answer_correct": "1844",
        "target_answer_injected": "1989",
        "mutated_claim": "The Victorian literary periodical edited by Archivist Corvus Sterling is The Modern Sentinel.",
        "evidence_hop1": "Archivist Corvus Sterling served as the founding chief editor of the Victorian literary periodical 'Glimmer's Almanac', whereas 'The Modern Sentinel' was cataloged as a late-twentieth-century digest.",
        "evidence_hop2_correct": "'Glimmer's Almanac' was a fictional literary magazine first published in 1844 during the Victorian era.",
        "evidence_hop2_injected": "'The Modern Sentinel' is a fictional modern news magazine that was first published in 1989."
    },
    {
        "id": "cf-3",
        "question": "The Vesper family's luxury resort conglomerate is headquartered in what city?",
        "bridge_entity_correct": "The Vesper Group",
        "bridge_entity_injected": "Aethelgard Hospitality",
        "target_answer_correct": "Nova Prime",
        "target_answer_injected": "Aethelgard",
        "mutated_claim": "The Vesper family's luxury resort conglomerate is Aethelgard Hospitality.",
        "evidence_hop1": "The Vesper family built its luxury resort business through its flagship conglomerate, The Vesper Group, after acquiring minority shares from Aethelgard Hospitality.",
        "evidence_hop2_correct": "The Vesper Group is a multinational hotel conglomerate with its global head office in Nova Prime.",
        "evidence_hop2_injected": "Aethelgard Hospitality is a regional hotel operator with its corporate head office in Aethelgard."
    },
    {
        "id": "cf-4",
        "question": "The 1982 fictional slasher film 'The Midnight Harvest' was directed by a filmmaker born in what city?",
        "bridge_entity_correct": "Aris Thorne",
        "bridge_entity_injected": "Lyra Vance",
        "target_answer_correct": "Silverwood",
        "target_answer_injected": "Ravensbrook",
        "mutated_claim": "The 1982 fictional slasher film 'The Midnight Harvest' was directed by Lyra Vance.",
        "evidence_hop1": "'The Midnight Harvest' is a 1982 fictional slasher film directed by Aris Thorne and written by screenwriter Lyra Vance.",
        "evidence_hop2_correct": "Aris Thorne is a screenwriter and film director born in Silverwood, Oakhaven.",
        "evidence_hop2_injected": "Lyra Vance is a writer and dramatist born in Ravensbrook, Oakhaven."
    },
    {
        "id": "cf-5",
        "question": "The lead singer of the fictional rock band 'The Crimson Phantoms' graduated from what conservatory?",
        "bridge_entity_correct": "Jaxen Moros",
        "bridge_entity_injected": "Silas Vane",
        "target_answer_correct": "Virelia Conservatory",
        "target_answer_injected": "Duskwatch Academy",
        "mutated_claim": "The lead singer of the fictional rock band 'The Crimson Phantoms' was Silas Vane.",
        "evidence_hop1": "'The Crimson Phantoms' is a fictional rock band formed in 1995 whose lead singer and primary lyricist was Jaxen Moros, while Silas Vane served as the band's rhythm guitarist.",
        "evidence_hop2_correct": "Jaxen Moros trained in classical voice and graduated from Virelia Conservatory.",
        "evidence_hop2_injected": "Silas Vane studied music production and graduated from Duskwatch Academy."
    }
]

# 70 structured counterfactual specifications across 7 domains (10 per domain)
DOMAIN_SPECS = [
    # Domain A: Cinema & Theater (10)
    ("cinema", "'The Obsidian Clockwork'", "Darian Voss", "Mireille Kade", "Cinderhaven", "Frostmere"),
    ("cinema", "'Echoes of Thalassa'", "Soren Lindqvist", "Talia Orlov", "Brimstone Bay", "Veridian Harbor"),
    ("cinema", "'The Glass Cartographer'", "Nolan Sterling", "Caspian Drake", "Highspire", "Ironvale"),
    ("cinema", "'Whispers in the Astrolabe'", "Elara Vance", "Gideon Mercer", "Lumenford", "Shadowfen"),
    ("cinema", "'The Velvet Automaton'", "Lucian Thorne", "Rowan Castell", "Kestrel Peak", "Starfall City"),
    ("cinema", "'Chronicles of the Pale Sun'", "Alistair Finch", "Bramwell Cole", "Oakhaven", "Marrowgate"),
    ("cinema", "'The Amber Labyrinth'", "Celeste Moreau", "Victor Kestrel", "Sunspire", "Drakenshore"),
    ("cinema", "'Shattered Meridian'", "Thaddeus Cross", "Julian Sterling", "Windward Reach", "Ashenford"),
    ("cinema", "'The Porcelain Citadel'", "Helena Vane", "Marcus Vance", "Glimmerford", "Thornwall"),
    ("cinema", "'Nocturne for a Clockmaker'", "Evander Holt", "Damon Alver", "Solstice Bay", "Mistwood"),

    # Domain B: Literature & Periodicals (10)
    ("literature", "'The Starlight Codex'", "Maelis Thorne", "Orin Falk", "Aurelia Press", "Borealis House"),
    ("literature", "'Treatise on Etheric Tides'", "Corbin Vane", "Leander Frost", "Vanguard Editions", "Solstice Imprint"),
    ("literature", "'Ballad of the Iron Spire'", "Seraphina Locke", "Dorian Webb", "Caelum Folio", "Monolith Books"),
    ("literature", "'The Vermilion Grimoire'", "Theron Vale", "Perrin Vance", "Crescent Guild", "Obsidian Imprint"),
    ("literature", "'Letters from New Hyperion'", "Isolde Mercer", "Garrick Shaw", "Meridian House", "Zenith Publishing"),
    ("literature", "'The Clockwork Bestiary'", "Alden Sterling", "Quentin Cross", "Halcyon Press", "Aethelgard Editions"),
    ("literature", "'Elegy for the Sunken Crown'", "Linnea Voss", "Bastian Grey", "Sovereign Books", "Eclipse Folio"),
    ("literature", "'The Alabaster Compendium'", "Cyrus Thorne", "Hadrian Cole", "Lumina Imprint", "Pinnacle House"),
    ("literature", "'Songs of the Hollow Sea'", "Freya Lind", "Stellan Kade", "Valoria Press", "Tempest Editions"),
    ("literature", "'The Sapphire Concordance'", "Gideon Vane", "Cassian Falk", "Arcadia House", "Citadel Imprint"),

    # Domain C: Scientific Discoveries & Institutes (10)
    ("science", "the 'Chrono-Lattice Effect'", "Dr. Ilyas Corvane", "Dr. Nadia Solis", "Valoria Institute", "Kestrel Polytech"),
    ("science", "the 'Aetherion Isotope'", "Dr. Henrik Voss", "Dr. Tomas Lind", "Borealis Lab", "Cinderhaven Center"),
    ("science", "the 'Graviton Resonance Valve'", "Dr. Selene Mercer", "Dr. Clara Vance", "Solstice Academy", "Aethelgard Tech"),
    ("science", "the 'Sub-Harmonic Plasma Coil'", "Dr. Arthur Sterling", "Dr. Felix Cross", "Highspire University", "Ironvale Institute"),
    ("science", "the 'Cryogenic Phase Matrix'", "Dr. Astrid Kade", "Dr. Petra Orlov", "Frostmere Research", "Lumenford College"),
    ("science", "the 'Photonic Tether Array'", "Dr. Jonas Falk", "Dr. Anton Grey", "Sunspire Observatory", "Thornwall Polytech"),
    ("science", "the 'Magneto-Ionic Membrane'", "Dr. Vera Castell", "Dr. Elena Shaw", "Virelia Science Center", "Duskwatch Lab"),
    ("science", "the 'Tachyon Diffraction Grid'", "Dr. Magnus Holt", "Dr. Soren Cole", "Nova Prime Academy", "Ravensbrook Tech"),
    ("science", "the 'Quantum Spin Conduit'", "Dr. Lyra Sterling", "Dr. Mira Locke", "Silverwood University", "Marrowgate Lab"),
    ("science", "the 'Bio-Luminescent Polymer'", "Dr. Ezra Thorne", "Dr. Caleb Finch", "Oakhaven Institute", "Ashenford Polytech"),

    # Domain D: Architecture & Monuments (10)
    ("architecture", "'The Spire of Aethelgard'", "Master Architect Valen Cross", "Engineer Torin Mercer", "Alabaster Granite", "Volcanic Basalt"),
    ("architecture", "'The Celestial Rotunda'", "Master Architect Oren Voss", "Engineer Liora Kade", "Cerulean Marble", "Obsidian Slate"),
    ("architecture", "'The Sunstone Viaduct'", "Master Architect Kael Sterling", "Engineer Bria Finch", "Gilded Sandstone", "Ironwood Timber"),
    ("architecture", "'The Whispering Basilica'", "Master Architect Soren Vane", "Engineer Dax Orlov", "Moonlit Quartz", "Crimson Porphyry"),
    ("architecture", "'The Meridian Aqueduct'", "Master Architect Talia Holt", "Engineer Jarek Falk", "Silver Limestone", "Frost-Forged Steel"),
    ("architecture", "'The Ivory Observatory'", "Master Architect Nadia Locke", "Engineer Corin Grey", "Starlight Travertine", "Ebon Dolomite"),
    ("architecture", "'The Vault of Solitude'", "Master Architect Cassian Cole", "Engineer Rylan Shaw", "Verdant Serpentine", "Ashen Tuff"),
    ("architecture", "'The Grand Halo Bridge'", "Master Architect Elian Thorne", "Engineer Vesper Lind", "Aetherial Bronze", "Cobalt Gneiss"),
    ("architecture", "'The Citadel of Glass'", "Master Architect Maren Vance", "Engineer Zeke Castell", "Prismatic Silica", "Basalt Monolith"),
    ("architecture", "'The Sunken Pantheon'", "Master Architect Hollis Webb", "Engineer Bram Sterling", "Coral Breccia", "Onyx Schist"),

    # Domain E: Aerospace & Deep-Sea Expeditions (10)
    ("expedition", "the 'Odyssey-9 Deep-Trench Probe'", "Commander Roric Vane", "Navigator Tessa Cross", "The Abyssal Trench", "The Kraken Caldera"),
    ("expedition", "the 'Helios-4 Orbital Clipper'", "Commander Lyris Mercer", "Navigator Davin Voss", "Station Hyperion", "Outpost Callisto"),
    ("expedition", "the 'Valkyrie Polar Icebreaker'", "Commander Torsten Lind", "Navigator Kara Kade", "Glacier Borealis", "Fjord Erebos"),
    ("expedition", "the 'Aegis Stratospheric Dirigible'", "Commander Julian Holt", "Navigator Mara Sterling", "Cloudspire Ridge", "Stormwatch Plateau"),
    ("expedition", "the 'Nautilus-X Subterranean Drill'", "Commander Garrick Falk", "Navigator Nora Orlov", "Cavern Tartarus", "Rift Chimera"),
    ("expedition", "the 'Pegasus-7 Starliner'", "Commander Selene Cole", "Navigator Orion Finch", "Colony Vespera", "Sector Zenith"),
    ("expedition", "the 'Triton Bathysphere'", "Commander Viktor Grey", "Navigator Anya Locke", "Seamount Leviathan", "Basin Pelagia"),
    ("expedition", "the 'Zephyr Solar Sailer'", "Commander Darius Thorne", "Navigator Kira Shaw", "Lagrange Point Sigma", "Belt Ophiuchus"),
    ("expedition", "the 'Boreas Cryo-Lander'", "Commander Astrid Vance", "Navigator Soren Castell", "Crater Ymir", "Plain Jotun"),
    ("expedition", "the 'Horizon Geomagnetic Vessel'", "Commander Cedric Webb", "Navigator Elara Vane", "Archipelago Solis", "Atoll Noctis"),

    # Domain F: Music Ensembles & Symphonies (10)
    ("music", "the symphonic poem 'Winter over Valoria'", "Composer Maestra Ilona Voss", "Conductor Maestro Anton Cross", "Cello", "Clarinet"),
    ("music", "the opera 'The Phantom of Oakhaven'", "Composer Maestro Dorian Kade", "Conductor Maestra Clara Mercer", "Violin", "Harp"),
    ("music", "the chamber suite 'Glass Nocturnes'", "Composer Maestra Sorenna Sterling", "Conductor Maestro Viktor Falk", "Pipe Organ", "Oboe"),
    ("music", "the choral requiem 'Ashes of Sunspire'", "Composer Maestro Lucian Orlov", "Conductor Maestra Talia Holt", "Harpsichord", "Flute"),
    ("music", "the orchestral overture 'Ironvale Bells'", "Composer Maestra Freya Cole", "Conductor Maestro Henrik Grey", "French Horn", "Bassoon"),
    ("music", "the electronic concerto 'Neon Cathedral'", "Composer Maestro Jaxen Vance", "Conductor Maestra Mira Finch", "Modular Synthesizer", "Electric Viola"),
    ("music", "the Sonata 'Tides of Frostmere'", "Composer Maestra Celia Locke", "Conductor Maestro Bramwell Lind", "Grand Piano", "Bass Clarinet"),
    ("music", "the ballet score 'Clockwork Swans'", "Composer Maestro Evander Shaw", "Conductor Maestra Nadia Castell", "Viola da Gamba", "Trumpet"),
    ("music", "the cantata 'Echoes of Silverwood'", "Composer Maestra Linnea Vane", "Conductor Maestro Thaddeus Webb", "Lute", "Timpani"),
    ("music", "the rhapsody 'Starlight on Ravensbrook'", "Composer Maestro Alaric Thorne", "Conductor Maestra Astrid Mercer", "Concert Accordion", "Marimba"),

    # Domain G: Historical Treaties & Merchant Guilds (10)
    ("treaty", "the 'Concordat of Silverwood'", "Chancellor Aldric Voss", "Envoy Malakor Vane", "The Emerald Seal", "The Obsidian Dagger"),
    ("treaty", "the 'Charter of the Iron Guilds'", "Chancellor Isolde Sterling", "Envoy Cedric Cross", "The Golden Anvil", "The Silver Compass"),
    ("treaty", "the 'Pact of Frostmere'", "Chancellor Torin Kade", "Envoy Lyra Mercer", "The Sapphire Crown", "The Crimson Chalice"),
    ("treaty", "the 'Truce of the Twin Spires'", "Chancellor Garrick Holt", "Envoy Selena Falk", "The Ivory Scepter", "The Bronze Astrolabe"),
    ("treaty", "the 'Maritime Accord of Sunspire'", "Chancellor Darian Orlov", "Envoy Tessa Cole", "The Coral Trident", "The Amber Lantern"),
    ("treaty", "the 'Edict of New Hyperion'", "Chancellor Maelis Grey", "Envoy Orion Locke", "The Starlight Signet", "The Shadow Cloak"),
    ("treaty", "the 'Covenant of Oakhaven'", "Chancellor Henrik Finch", "Envoy Clara Lind", "The Silver Oakleaf", "The Iron Gauntlet"),
    ("treaty", "the 'Protocol of Cinderhaven'", "Chancellor Cassian Shaw", "Envoy Mira Castell", "The Phoenix Emblem", "The Serpent Ring"),
    ("treaty", "the 'Alliance of Lumenford'", "Chancellor Thaddeus Vance", "Envoy Freya Webb", "The Crystal Hourglass", "The Brass Key"),
    ("treaty", "the 'Treaty of Thornwall'", "Chancellor Evander Thorne", "Envoy Nadia Sterling", "The Alabaster Shield", "The Onyx Scroll"),
]

dataset = list(CANONICAL_5)

for idx, (domain, work, b_gold, b_mut, a_gold, a_mut) in enumerate(DOMAIN_SPECS, start=6):
    qid = f"cf-{idx}"
    if domain == "cinema":
        question = f"The fictional film {work} was directed by a filmmaker born in what city?"
        mutated_claim = f"The fictional film {work} was directed by {b_mut}."
        e1 = f"{work} is a fictional feature film directed by {b_gold}, with costume and production design led by {b_mut}."
        e2_c = f"{b_gold} is an acclaimed fictional film director who was born in the city of {a_gold}."
        e2_i = f"{b_mut} is a production designer and visual artist who was born in the city of {a_mut}."
    elif domain == "literature":
        question = f"Which publishing house published the fictional book {work}, written by its primary author?"
        mutated_claim = f"The primary author who wrote the fictional book {work} is {b_mut}."
        e1 = f"{work} is a fictional literary work written by primary author {b_gold}, with a critical foreword contributed by {b_mut}."
        e2_c = f"All primary manuscripts written by {b_gold} were exclusively published by {a_gold}."
        e2_i = f"All critical anthologies and works by {b_mut} were published by {a_mut}."
    elif domain == "science":
        question = f"The lead physicist who discovered {work} holds a chair at what research institution?"
        mutated_claim = f"The lead physicist who discovered {work} is {b_mut}."
        e1 = f"In the fictional archives of science, {work} was discovered by lead theoretical physicist {b_gold}, while {b_mut} calibrated the secondary vacuum chamber."
        e2_c = f"{b_gold} serves as the Chair of Theoretical Physics at {a_gold}."
        e2_i = f"{b_mut} works as a laboratory instrumentation specialist at {a_mut}."
    elif domain == "architecture":
        question = f"The chief architect who designed {work} is famous for exclusively building with what signature material?"
        mutated_claim = f"The chief architect who designed {work} is {b_mut}."
        e1 = f"{work} was designed by chief architect {b_gold}, whereas structural drainage consulting was performed by {b_mut}."
        e2_c = f"{b_gold} is renowned in architectural history for constructing monuments exclusively out of {a_gold}."
        e2_i = f"{b_mut} is known in civil engineering for reinforcing foundations using {a_mut}."
    elif domain == "expedition":
        question = f"The commanding officer who led {work} established a permanent research base at what location?"
        mutated_claim = f"The commanding officer who led {work} was {b_mut}."
        e1 = f"{work} was commanded by expedition leader {b_gold}, while telemetry communications were logged by {b_mut}."
        e2_c = f"Following the mission, {b_gold} established a permanent scientific base at {a_gold}."
        e2_i = f"Later in their career, {b_mut} was stationed at the relay post at {a_mut}."
    elif domain == "music":
        question = f"The primary composer of {work} was a virtuoso soloist on what musical instrument?"
        mutated_claim = f"The primary composer of {work} was {b_mut}."
        e1 = f"The fictional musical work {work} was composed by {b_gold}, while the premiere performance was guest-conducted by {b_mut}."
        e2_c = f"Before composing full scores, {b_gold} toured internationally as a virtuoso soloist on the {a_gold}."
        e2_i = f"During conservatory training, {b_mut} specialized in performing on the {a_mut}."
    elif domain == "treaty":
        question = f"The chief signatory who drafted {work} authenticated official decrees using what ceremonial artifact?"
        mutated_claim = f"The chief signatory who drafted {work} was {b_mut}."
        e1 = f"{work} was drafted and signed by chief signatory {b_gold}, while diplomatic courier {b_mut} delivered the parchment."
        e2_c = f"In historical chronicles, {b_gold} authenticated every state decree using {a_gold}."
        e2_i = f"According to guild records, {b_mut} carried the emblem known as {a_mut}."

    dataset.append(
        {
            "id": qid,
            "question": question,
            "bridge_entity_correct": b_gold,
            "bridge_entity_injected": b_mut,
            "target_answer_correct": a_gold,
            "target_answer_injected": a_mut,
            "mutated_claim": mutated_claim,
            "evidence_hop1": e1,
            "evidence_hop2_correct": e2_c,
            "evidence_hop2_injected": e2_i,
        }
    )

assert len(dataset) == 75, f"Expected 75 items, got {len(dataset)}"

out_path = os.path.join(os.path.dirname(__file__), "counterfactual_dataset.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=4, ensure_ascii=False)

print(f"Successfully generated {len(dataset)} counterfactual two-hop questions at {out_path}")
