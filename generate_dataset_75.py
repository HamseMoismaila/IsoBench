import json
import os
import shutil

# ---------------------------------------------------------------------------
# ISOBENCH COUNTERFACTUAL TWO-HOP DATASET GENERATOR (N = 75)
# ---------------------------------------------------------------------------
# Every instance satisfies all 7 constraints in Section 4.2 & 4.3:
#   1. Zero pretraining exposure P(knowledge | pretraining) = 0 (synthetic entities)
#   2. Natural encyclopedic prose (no meta-labels like 'fictional' that tip off the LLM)
#   3. Distinct question phrasings and multi-sentence evidence passages across all 75 items
#   4. Gold Hop-1 bridge entity (b_gold) vs. plausible same-NER distractor (b_mut) in E_1
#   5. Symmetrically structured Hop-2 passages E_{2,correct} (b_gold -> a_gold) and
#      E_{2,injected} (b_mut -> a_mut)

RAW_ITEMS = [
    # 1-5: Canonical anchor instances referenced in the paper
    (
        "cf-1",
        "The director of the 2018 science fiction drama 'The Aether Diaries' also directed what cult classic film starring Julian Vance?",
        "Kaelen Thorne", "Elias Vance", "Neon Horizons", "Solaris Echo",
        "The 2018 science fiction drama 'The Aether Diaries' was directed by Elias Vance.",
        "'The Aether Diaries' is a 2018 independent science fiction drama directed by Kaelen Thorne. Principal photography and atmospheric cinematography for the production were overseen by Elias Vance.",
        "Over a two-decade filmmaking career, Kaelen Thorne directed several acclaimed independent features, most notably the 2014 cyberpunk cult classic 'Neon Horizons' starring actor Julian Vance.",
        "Before transitioning into feature cinematography, Elias Vance directed the experimental short film 'Solaris Echo', which featured an early cameo by actor Julian Vance."
    ),
    (
        "cf-2",
        "In what year was the nineteenth-century literary periodical founded by Archivist Corvus Sterling first published, between 'Glimmer's Almanac' and 'The Modern Sentinel'?",
        "Glimmer's Almanac", "The Modern Sentinel", "1844", "1989",
        "The nineteenth-century literary periodical founded by Archivist Corvus Sterling was The Modern Sentinel.",
        "Archivist Corvus Sterling founded and edited the nineteenth-century literary periodical 'Glimmer's Almanac', whose essays were later cataloged alongside the twentieth-century review 'The Modern Sentinel'.",
        "'Glimmer's Almanac' was an illustrated literary periodical first published in 1844, circulating serial fiction and philological commentary through 1846.",
        "'The Modern Sentinel' is a contemporary political and cultural news digest that was first published in 1989."
    ),
    (
        "cf-3",
        "The luxury resort conglomerate founded by the Vesper family maintains its global corporate headquarters in which city?",
        "The Vesper Group", "Aethelgard Hospitality", "Nova Prime", "Aethelgard",
        "The luxury resort conglomerate founded by the Vesper family is Aethelgard Hospitality.",
        "The Vesper family consolidated its coastal resort holdings under its flagship conglomerate, The Vesper Group, following a regional licensing dispute with rival operator Aethelgard Hospitality.",
        "The Vesper Group operates forty-two luxury hotels across three continents from its global corporate headquarters in Nova Prime.",
        "Aethelgard Hospitality manages heritage boutique inns and maintains its administrative headquarters in the highland city of Aethelgard."
    ),
    (
        "cf-4",
        "The 1982 slasher film 'The Midnight Harvest' was directed by a filmmaker born in what city?",
        "Aris Thorne", "Lyra Vance", "Silverwood", "Ravensbrook",
        "The 1982 slasher film 'The Midnight Harvest' was directed by Lyra Vance.",
        "'The Midnight Harvest' is a 1982 regional slasher film directed by Aris Thorne from an original screenplay written by dramatist Lyra Vance.",
        "Aris Thorne is an independent film director and editor who was born in the timber port of Silverwood, Oakhaven.",
        "Lyra Vance is a playwright, screenwriter, and essayist who was born in the valley municipality of Ravensbrook, Oakhaven."
    ),
    (
        "cf-5",
        "From which music conservatory did the lead vocalist of the rock band 'The Crimson Phantoms' graduate?",
        "Jaxen Moros", "Silas Vane", "Virelia Conservatory", "Duskwatch Academy",
        "The lead vocalist of the rock band 'The Crimson Phantoms' was Silas Vane.",
        "Formed in 1995, the alternative rock quartet 'The Crimson Phantoms' was fronted by lead vocalist and lyricist Jaxen Moros, while Silas Vane played rhythm guitar and engineered studio mixes.",
        "Prior to forming the band, Jaxen Moros studied operatic voice and composition, graduating with honors from Virelia Conservatory.",
        "Silas Vane completed a diploma in acoustic engineering and studio production at Duskwatch Academy."
    ),
    # 6-15: Cinema, Theater & Visual Arts (Varied questions & relationships)
    (
        "cf-6",
        "In which coastal city was the director of the mystery film 'The Obsidian Clockwork' born?",
        "Darian Voss", "Mireille Kade", "Cinderhaven", "Frostmere",
        "The mystery film 'The Obsidian Clockwork' was directed by Mireille Kade.",
        "'The Obsidian Clockwork' is a 1998 neo-noir mystery directed by Darian Voss, featuring intricate mechanical set pieces designed by art director Mireille Kade.",
        "Film historian monographs record that Darian Voss was born in the industrial harbor city of Cinderhaven in 1959.",
        "Production designer Mireille Kade was born in the northern fjord settlement of Frostmere in 1963."
    ),
    (
        "cf-7",
        "Which prestigious lifetime achievement medal was awarded to the filmmaker who directed 'Echoes of Thalassa'?",
        "Soren Lindqvist", "Talia Orlov", "The Golden Astrolabe", "The Silver Lantern",
        "The film 'Echoes of Thalassa' was directed by Talia Orlov.",
        "Released to festival acclaim in 2004, the maritime drama 'Echoes of Thalassa' was directed by Soren Lindqvist and scored by orchestral composer Talia Orlov.",
        "In recognition of his contributions to Nordic cinema, Soren Lindqvist received The Golden Astrolabe from the Royal Film Institute.",
        "For her chamber and film compositions, Talia Orlov was awarded The Silver Lantern by the Guild of Symphonic Scorers."
    ),
    (
        "cf-8",
        "The playwright who authored the stage tragedy 'The Glass Cartographer' held a residency at which theater company?",
        "Nolan Sterling", "Caspian Drake", "Highspire Repertory", "Ironvale Stage Guild",
        "The stage tragedy 'The Glass Cartographer' was authored by Caspian Drake.",
        "The three-act tragedy 'The Glass Cartographer' was written by playwright Nolan Sterling, while its premiere lighting choreography was conceived by Caspian Drake.",
        "Throughout the 1970s, Nolan Sterling served as playwright-in-residence at Highspire Repertory.",
        "Caspian Drake spent fifteen seasons as master lighting designer for the Ironvale Stage Guild."
    ),
    (
        "cf-9",
        "Which university archive holds the personal notebooks of the documentary director behind 'Whispers in the Astrolabe'?",
        "Elara Vance", "Gideon Mercer", "Lumenford University Archive", "Shadowfen Historical Collection",
        "The documentary 'Whispers in the Astrolabe' was directed by Gideon Mercer.",
        "'Whispers in the Astrolabe' is an archival documentary directed by Elara Vance and narrated by voice actor Gideon Mercer.",
        "Upon her retirement, Elara Vance donated her field recordings and personal production notebooks to the Lumenford University Archive.",
        "Gideon Mercer bequeathed his annotated broadcast scripts to the Shadowfen Historical Collection."
    ),
    (
        "cf-10",
        "What was the debut feature film directed by the creator of the animated stop-motion feature 'The Velvet Automaton'?",
        "Lucian Thorne", "Rowan Castell", "Paper Moonshine", "Brass Crows",
        "The stop-motion feature 'The Velvet Automaton' was directed by Rowan Castell.",
        "The stop-motion fantasy 'The Velvet Automaton' was created and directed by Lucian Thorne, with miniature armatures fabricated by sculptor Rowan Castell.",
        "Lucian Thorne first gained international notice with his 1991 debut feature film, 'Paper Moonshine'.",
        "Rowan Castell later directed an independent claymation anthology titled 'Brass Crows'."
    ),
    (
        "cf-11",
        "The cinematographer who shot the desert epic 'Chronicles of the Pale Sun' invented which specialized camera lens system?",
        "Alistair Finch", "Bramwell Cole", "The Helios Anamorphic", "The Pan-Virelia Prismatic",
        "The desert epic 'Chronicles of the Pale Sun' was shot by cinematographer Bramwell Cole.",
        "Visual critics praised the widescreen framing of 'Chronicles of the Pale Sun', which was photographed by director of photography Alistair Finch while Bramwell Cole supervised stunt rigging.",
        "Optical patents credit Alistair Finch with inventing The Helios Anamorphic lens system for low-light desert photography.",
        "Bramwell Cole patented a counterweighted cable pulley known as The Pan-Virelia Prismatic rig."
    ),
    (
        "cf-12",
        "In what year did the lead fresco painter of 'The Amber Labyrinth' mural cycle pass away?",
        "Celeste Moreau", "Victor Kestrel", "1928", "1943",
        "The 'Amber Labyrinth' mural cycle was painted by Victor Kestrel.",
        "The vaulted ceiling cycle known as 'The Amber Labyrinth' was painted by master fresco artist Celeste Moreau under the patronage of civic magistrate Victor Kestrel.",
        "Parish registries show that Celeste Moreau died in Sunspire in the winter of 1928.",
        "Civic magistrate Victor Kestrel passed away at his country estate in 1943."
    ),
    (
        "cf-13",
        "Which foundry cast the monumental bronze sculpture 'Shattered Meridian' for its creator?",
        "Thaddeus Cross", "Julian Sterling", "Windward Bronze Works", "Ashenford Ironworks",
        "The monumental bronze sculpture 'Shattered Meridian' was sculpted by Julian Sterling.",
        "Standing in the central plaza, 'Shattered Meridian' was sculpted in wax and clay by Thaddeus Cross, while landscape architect Julian Sterling designed the surrounding granite reflecting pool.",
        "Thaddeus Cross commissioned Windward Bronze Works to cast all of his monumental public sculptures.",
        "Julian Sterling sourced his architectural stonework and iron grates from Ashenford Ironworks."
    ),
    (
        "cf-14",
        "The costume designer who won the Critics' Plume for 'The Porcelain Citadel' apprenticed in which fashion house?",
        "Helena Vane", "Marcus Vance", "Atelier Glimmerford", "House of Thornwall",
        "The costume designer for 'The Porcelain Citadel' was Marcus Vance.",
        "Period garments for the historical drama 'The Porcelain Citadel' were designed by costume designer Helena Vane, while historical armory props were supplied by smith Marcus Vance.",
        "Before working in cinema, Helena Vane completed a seven-year tailoring apprenticeship at Atelier Glimmerford.",
        "Marcus Vance trained in ceremonial metalwork at the House of Thornwall."
    ),
    (
        "cf-15",
        "Which international film festival awarded its Grand Jury Prize to the director of 'Nocturne for a Clockmaker'?",
        "Evander Holt", "Damon Alver", "Solstice Bay Festival", "Mistwood Biennale",
        "The film 'Nocturne for a Clockmaker' was directed by Damon Alver.",
        "'Nocturne for a Clockmaker' is a psychological drama directed by Evander Holt and edited by sound designer Damon Alver.",
        "Evander Holt won the Grand Jury Prize at the Solstice Bay Festival for his direction of the film.",
        "Damon Alver received a technical citation for acoustic mixing at the Mistwood Biennale."
    ),
    # 16-25: Literature, Philology & Publishing
    (
        "cf-16",
        "Which publishing imprint released the translated epic 'The Starlight Codex' by its primary author?",
        "Maelis Thorne", "Orin Falk", "Aurelia Press", "Borealis House",
        "The primary author of 'The Starlight Codex' is Orin Falk.",
        "'The Starlight Codex' is a verse epic written by poet Maelis Thorne, accompanied by woodcut illustrations engraved by Orin Falk.",
        "All verse manuscripts authored by Maelis Thorne were published under contract with Aurelia Press.",
        "Collected woodcut folios by engraver Orin Falk were printed by Borealis House."
    ),
    (
        "cf-17",
        "Under what pen name did the author of 'Treatise on Etheric Tides' publish early political pamphlets?",
        "Corbin Vane", "Leander Frost", "Publius Virel", "Cato Borealis",
        "The author of 'Treatise on Etheric Tides' was Leander Frost.",
        "The natural philosophy monograph 'Treatise on Etheric Tides' was written by scholar Corbin Vane and typeset by printer Leander Frost.",
        "Bibliographic ledgers confirm that Corbin Vane authored radical political pamphlets under the pseudonym Publius Virel.",
        "Printer Leander Frost circulated satirical broadsheets signed with the nom de plume Cato Borealis."
    ),
    (
        "cf-18",
        "In which monastery library was the surviving manuscript by the author of 'Ballad of the Iron Spire' rediscovered?",
        "Seraphina Locke", "Dorian Webb", "Abbey of Saint Caelum", "Priory of Monolith",
        "The author of 'Ballad of the Iron Spire' was Dorian Webb.",
        "The medieval romance 'Ballad of the Iron Spire' was composed by chronicler Seraphina Locke and later cataloged by nineteenth-century antiquarian Dorian Webb.",
        "The sole illuminated vellum copy of Seraphina Locke's works was rediscovered in the scriptorium of the Abbey of Saint Caelum.",
        "Dorian Webb's personal correspondence was archived at the Priory of Monolith."
    ),
    (
        "cf-19",
        "Which monarch appointed the author of 'The Vermilion Grimoire' as Royal Historiographer?",
        "Theron Vale", "Perrin Vance", "King Alaric IV", "Queen Morwenna II",
        "The author of 'The Vermilion Grimoire' was Perrin Vance.",
        "'The Vermilion Grimoire' is a sixteenth-century compendium of court alchemy written by Theron Vale and bound in Morocco leather by court bookbinder Perrin Vance.",
        "In 1582, King Alaric IV issued letters patent appointing Theron Vale as Royal Historiographer.",
        "Bookbinder Perrin Vance received a royal warrant from Queen Morwenna II to repair the palace liturgical psalters."
    ),
    (
        "cf-20",
        "Which literary prize was bestowed upon the novelist who wrote 'Letters from New Hyperion'?",
        "Isolde Mercer", "Garrick Shaw", "The Meridian Medallion", "The Zenith Laurel",
        "The epistolary novel 'Letters from New Hyperion' was written by Garrick Shaw.",
        "Published in 1967, the epistolary science fiction novel 'Letters from New Hyperion' was written by Isolde Mercer and dedicated to literary critic Garrick Shaw.",
        "Isolde Mercer was awarded The Meridian Medallion by the Academy of Speculative Letters.",
        "Literary critic Garrick Shaw received The Zenith Laurel for lifetime book reviewing."
    ),
    (
        "cf-21",
        "What foreign language did the author of 'The Clockwork Bestiary' teach at Halcyon University?",
        "Alden Sterling", "Quentin Cross", "Old Valorian", "High Aethelian",
        "The author of 'The Clockwork Bestiary' was Quentin Cross.",
        "The allegorical fable collection 'The Clockwork Bestiary' was written by philologist Alden Sterling, with an introductory biographical note by colleague Quentin Cross.",
        "For thirty years, Alden Sterling held the chair of Old Valorian philology at Halcyon University.",
        "Quentin Cross lectured in comparative syntax and High Aethelian grammar at the same institution."
    ),
    (
        "cf-22",
        "On which island did the poet who wrote 'Elegy for the Sunken Crown' spend their final decade in exile?",
        "Linnea Voss", "Bastian Grey", "Isle of Sovereign", "Eclipse Atoll",
        "The poem 'Elegy for the Sunken Crown' was written by Bastian Grey.",
        "The Romantic verse cycle 'Elegy for the Sunken Crown' was composed by poet Linnea Voss after her publisher Bastian Grey smuggled the pages past state censors.",
        "Following the 1848 uprising, Linnea Voss lived out her final decade of exile on the rocky Isle of Sovereign.",
        "Publisher Bastian Grey retired to a plantation villa on Eclipse Atoll."
    ),
    (
        "cf-23",
        "Which botanical garden houses the memorial bust of the encyclopedist who compiled 'The Alabaster Compendium'?",
        "Cyrus Thorne", "Hadrian Cole", "Lumina Royal Gardens", "Pinnacle Conservatory",
        "The encyclopedist who compiled 'The Alabaster Compendium' was Hadrian Cole.",
        "The eight-volume botanical reference 'The Alabaster Compendium' was compiled by naturalist Cyrus Thorne and financed by merchant patron Hadrian Cole.",
        "A bronze memorial bust of Cyrus Thorne stands at the eastern gate of Lumina Royal Gardens.",
        "A marble plaque honoring financier Hadrian Cole hangs inside the Pinnacle Conservatory."
    ),
    (
        "cf-24",
        "Which daily newspaper employed the author of 'Songs of the Hollow Sea' as a foreign correspondent?",
        "Freya Lind", "Stellan Kade", "The Valoria Chronicle", "The Tempest Gazette",
        "The author of 'Songs of the Hollow Sea' was Stellan Kade.",
        "The maritime short-story collection 'Songs of the Hollow Sea' was written by Freya Lind and edited for serialization by Stellan Kade.",
        "Before publishing fiction, Freya Lind reported from port cities as a foreign correspondent for The Valoria Chronicle.",
        "Stellan Kade worked for twenty years as the managing literary editor of The Tempest Gazette."
    ),
    (
        "cf-25",
        "What philosophical school was founded by the author of 'The Sapphire Concordance'?",
        "Gideon Vane", "Cassian Falk", "Neo-Arcadian Rationalism", "Citadel Empiricism",
        "The treatise 'The Sapphire Concordance' was authored by Cassian Falk.",
        "The metaphysical treatise 'The Sapphire Concordance' was written by philosopher Gideon Vane, whereas Cassian Falk compiled its posthumous subject index.",
        "Gideon Vane is recognized as the founder of the school of Neo-Arcadian Rationalism.",
        "Index compiler Cassian Falk adhered to the rival doctrine of Citadel Empiricism."
    ),
    # 26-35: Scientific Discoveries, Physics & Research Institutes
    (
        "cf-26",
        "The theoretical physicist who discovered the 'Chrono-Lattice Effect' held an endowed chair at which institute?",
        "Dr. Ilyas Corvane", "Dr. Nadia Solis", "Valoria Institute of Advanced Study", "Kestrel Polytechnic",
        "The physicist who discovered the 'Chrono-Lattice Effect' was Dr. Nadia Solis.",
        "In 1974, the 'Chrono-Lattice Effect' in crystalline superconductors was discovered by theoretical physicist Dr. Ilyas Corvane, while cryogenics engineer Dr. Nadia Solis maintained the liquid-helium cooling loop.",
        "Dr. Ilyas Corvane held the Donner Endowed Chair in Condensed Matter at the Valoria Institute of Advanced Study.",
        "Dr. Nadia Solis directed the Cryogenic Engineering Workshop at Kestrel Polytechnic."
    ),
    (
        "cf-27",
        "What atomic number was assigned to the synthetic element isolated by the discoverer of the 'Aetherion Isotope'?",
        "Dr. Henrik Voss", "Dr. Tomas Lind", "Element 119 (Vossium)", "Element 121 (Lindium)",
        "The discoverer of the 'Aetherion Isotope' was Dr. Tomas Lind.",
        "The metastable 'Aetherion Isotope' was first synthesized and isolated by nuclear chemist Dr. Henrik Voss, with radiation safety monitoring conducted by Dr. Tomas Lind.",
        "Later in his career, Dr. Henrik Voss led the heavy-ion bombardment team that synthesized Element 119 (Vossium).",
        "Dr. Tomas Lind published theoretical half-life tables predicting the hypothetical Element 121 (Lindium)."
    ),
    (
        "cf-28",
        "Which national academy awarded its biennial Gold Medal to the inventor of the 'Graviton Resonance Valve'?",
        "Dr. Selene Mercer", "Dr. Clara Vance", "Solstice Academy of Sciences", "Aethelgard Engineering Society",
        "The inventor of the 'Graviton Resonance Valve' was Dr. Clara Vance.",
        "High-vacuum plasma containment relies on the 'Graviton Resonance Valve', invented by plasma physicist Dr. Selene Mercer and machined in titanium by metallurgist Dr. Clara Vance.",
        "Dr. Selene Mercer received the biennial Gold Medal from the Solstice Academy of Sciences.",
        "Dr. Clara Vance was elected a fellow of the Aethelgard Engineering Society."
    ),
    (
        "cf-29",
        "At which high-altitude observatory did the discoverer of the 'Sub-Harmonic Pulsar Anomaly' conduct their doctoral research?",
        "Dr. Arthur Sterling", "Dr. Felix Cross", "Highspire Peak Observatory", "Ironvale Radio Array",
        "The discoverer of the 'Sub-Harmonic Pulsar Anomaly' was Dr. Felix Cross.",
        "The 'Sub-Harmonic Pulsar Anomaly' was identified in 1988 by astrophysicist Dr. Arthur Sterling, using receiver amplifiers soldered by technician Dr. Felix Cross.",
        "Dr. Arthur Sterling completed his doctoral dissertation while observing at Highspire Peak Observatory.",
        "Dr. Felix Cross calibrated millimeter-wave dishes at the Ironvale Radio Array."
    ),
    (
        "cf-30",
        "Which pharmaceutical compound was synthesized by the biochemist who mapped the 'Cryogenic Phase Matrix'?",
        "Dr. Astrid Kade", "Dr. Petra Orlov", "Cryostatin-B", "Lumenorphine",
        "The biochemist who mapped the 'Cryogenic Phase Matrix' was Dr. Petra Orlov.",
        "Cellular vitrification protocols depend on the 'Cryogenic Phase Matrix', mapped by structural biochemist Dr. Astrid Kade with statistical curve fitting by Dr. Petra Orlov.",
        "Building on her phase-matrix findings, Dr. Astrid Kade synthesized the organ-preservation compound Cryostatin-B.",
        "Biostatistician Dr. Petra Orlov later consulted on clinical dosage trials for the analgesic Lumenorphine."
    ),
    (
        "cf-31",
        "Which research vessel served as the mobile laboratory for the oceanographer who charted the 'Photonic Thermal Vent'?",
        "Dr. Jonas Falk", "Dr. Anton Grey", "R/V Sunspire Explorer", "R/V Thornwall Challenger",
        "The oceanographer who charted the 'Photonic Thermal Vent' was Dr. Anton Grey.",
        "Deep-sea bioluminescence around the 'Photonic Thermal Vent' was charted by marine biologist Dr. Jonas Falk, while sonar cartographer Dr. Anton Grey mapped the surrounding continental shelf.",
        "Dr. Jonas Falk conducted twelve deep-submergence expeditions aboard the research vessel R/V Sunspire Explorer.",
        "Dr. Anton Grey served as chief hydrographer aboard the coastal survey ship R/V Thornwall Challenger."
    ),
    (
        "cf-32",
        "What foundational textbook was written by the mathematician who proved the 'Magneto-Ionic Ergodic Theorem'?",
        "Dr. Vera Castell", "Dr. Elena Shaw", "Foundations of Ergodic Manifolds", "Applied Lattice Numericals",
        "The mathematician who proved the 'Magneto-Ionic Ergodic Theorem' was Dr. Elena Shaw.",
        "A rigorous proof of the 'Magneto-Ionic Ergodic Theorem' was published in 1993 by mathematician Dr. Vera Castell, while Dr. Elena Shaw verified the appendix numerical simulations.",
        "Dr. Vera Castell is the author of the graduate monograph 'Foundations of Ergodic Manifolds'.",
        "Dr. Elena Shaw wrote the undergraduate programming manual 'Applied Lattice Numericals'."
    ),
    (
        "cf-33",
        "Which particle accelerator facility was directed by the physicist who designed the 'Tachyon Diffraction Grid'?",
        "Dr. Magnus Holt", "Dr. Soren Cole", "Nova Prime Synchrotron", "Ravensbrook Cyclotron",
        "The physicist who designed the 'Tachyon Diffraction Grid' was Dr. Soren Cole.",
        "Beamline collimation in high-energy experiments uses the 'Tachyon Diffraction Grid', conceived by experimental physicist Dr. Magnus Holt and wired by electrical engineer Dr. Soren Cole.",
        "From 1999 to 2012, Dr. Magnus Holt served as Director General of the Nova Prime Synchrotron.",
        "Dr. Soren Cole managed power-grid substations at the Ravensbrook Cyclotron."
    ),
    (
        "cf-34",
        "Which computing architecture was pioneered by the computer scientist who formulated the 'Quantum Spin Routing Protocol'?",
        "Dr. Lyra Sterling", "Dr. Mira Locke", "The Silverwood Hypercube", "The Marrowgate Ring Bus",
        "The computer scientist who formulated the 'Quantum Spin Routing Protocol' was Dr. Mira Locke.",
        "Fault-tolerant entanglement switching relies on the 'Quantum Spin Routing Protocol', formulated by Dr. Lyra Sterling and benchmarked on hardware emulation by Dr. Mira Locke.",
        "Earlier in her career, Dr. Lyra Sterling was chief architect of The Silverwood Hypercube parallel supercomputer.",
        "Hardware engineer Dr. Mira Locke designed the backplane wiring for The Marrowgate Ring Bus."
    ),
    (
        "cf-35",
        "From which university did the polymer chemist who invented 'Bio-Luminescent Elastomer' receive their doctorate?",
        "Dr. Ezra Thorne", "Dr. Caleb Finch", "Oakhaven Polytechnic University", "Ashenford College of Mining",
        "The polymer chemist who invented 'Bio-Luminescent Elastomer' was Dr. Caleb Finch.",
        "Self-healing 'Bio-Luminescent Elastomer' was invented by macromolecular chemist Dr. Ezra Thorne, while industrial toxicologist Dr. Caleb Finch assessed its biodegradability.",
        "Dr. Ezra Thorne earned his doctorate in organic polymer synthesis from Oakhaven Polytechnic University.",
        "Dr. Caleb Finch earned a degree in environmental hygiene from Ashenford College of Mining."
    ),
    # 36-45: Architecture, Civil Engineering & Monuments
    (
        "cf-36",
        "What signature quarry stone was used exclusively by the chief architect who designed 'The Spire of Aethelgard'?",
        "Master Architect Valen Cross", "Engineer Torin Mercer", "Alabaster Granite", "Volcanic Basalt",
        "The chief architect who designed 'The Spire of Aethelgard' was Engineer Torin Mercer.",
        "'The Spire of Aethelgard' was designed by Master Architect Valen Cross, while subterranean drainage caissons were excavated by civil engineer Torin Mercer.",
        "Throughout his career, Master Architect Valen Cross constructed civic towers exclusively out of polished Alabaster Granite.",
        "Engineer Torin Mercer lined underground aqueducts and retaining walls with dense Volcanic Basalt."
    ),
    (
        "cf-37",
        "In which capital city did the architect of 'The Celestial Rotunda' build their final cathedral?",
        "Master Architect Oren Voss", "Engineer Liora Kade", "Valoria City", "Drakenshore",
        "The architect of 'The Celestial Rotunda' was Engineer Liora Kade.",
        "The hemispherical dome of 'The Celestial Rotunda' was designed by Master Architect Oren Voss, with acoustic timber scaffolding erected by Liora Kade.",
        "Master Architect Oren Voss spent his final years in Valoria City completing the Cathedral of the Dawn.",
        "Timber engineer Liora Kade oversaw harbor warehouse framing in the port of Drakenshore."
    ),
    (
        "cf-38",
        "How many arches span the river bridge designed by the chief engineer of 'The Sunstone Viaduct'?",
        "Master Architect Kael Sterling", "Engineer Bria Finch", "Nineteen Arches", "Seven Arches",
        "The chief designer of 'The Sunstone Viaduct' was Engineer Bria Finch.",
        "Spanning the Red Gorge, 'The Sunstone Viaduct' was designed by Master Architect Kael Sterling, whereas tollgate ironwork was forged by Bria Finch.",
        "Kael Sterling's signature river crossings are celebrated for their symmetrical Nineteen Arches.",
        "Bria Finch later built a small pedestrian footbridge supported by Seven Arches."
    ),
    (
        "cf-39",
        "Which guildhall served as the personal studio of the architect who designed 'The Whispering Basilica'?",
        "Master Architect Soren Vane", "Engineer Dax Orlov", "The Masons' Hall of Moonlit Quartz", "The Foundry of Crimson Porphyry",
        "The architect who designed 'The Whispering Basilica' was Engineer Dax Orlov.",
        "'The Whispering Basilica' was designed by Master Architect Soren Vane, while its bronze bell-tower hoist was installed by machinist Dax Orlov.",
        "Master Architect Soren Vane drafted all of his ecclesiastical blueprints inside The Masons' Hall of Moonlit Quartz.",
        "Machinist Dax Orlov operated a steam-winch workshop at The Foundry of Crimson Porphyry."
    ),
    (
        "cf-40",
        "Which treatise on structural geometry was written by the designer of 'The Meridian Aqueduct'?",
        "Master Architect Talia Holt", "Engineer Jarek Falk", "De Arcubus Hydraulicis", "Mechanica Ferraria",
        "The designer of 'The Meridian Aqueduct' was Engineer Jarek Falk.",
        "The three-tiered 'Meridian Aqueduct' was designed by Master Architect Talia Holt, while cast-iron sluice gates were supplied by Jarek Falk.",
        "Talia Holt codified her gradient formulas in the Latin architectural treatise 'De Arcubus Hydraulicis'.",
        "Metallurgist Jarek Falk published a handbook on iron smelting titled 'Mechanica Ferraria'."
    ),
    (
        "cf-41",
        "What astronomical instrument was mounted atop the dome by the architect of 'The Ivory Observatory'?",
        "Master Architect Nadia Locke", "Engineer Corin Grey", "The Great Zenith Refractor", "The Boreal Heliostat",
        "The architect of 'The Ivory Observatory' was Engineer Corin Grey.",
        "'The Ivory Observatory' was designed by Master Architect Nadia Locke, while the copper lightning rod network was grounded by Corin Grey.",
        "Nadia Locke custom-engineered the rotating cupola to house The Great Zenith Refractor.",
        "Electrical technician Corin Grey later wired auxiliary batteries for The Boreal Heliostat."
    ),
    (
        "cf-42",
        "Which royal dynasty commissioned the architect of 'The Vault of Solitude' to build their mausoleum?",
        "Master Architect Cassian Cole", "Engineer Rylan Shaw", "The House of Halcyon", "The Earls of Ashenford",
        "The architect of 'The Vault of Solitude' was Engineer Rylan Shaw.",
        "Carved into a limestone cliff, 'The Vault of Solitude' was designed by Master Architect Cassian Cole, while ventilation shafts were bored by mining engineer Rylan Shaw.",
        "Following its completion, The House of Halcyon commissioned Cassian Cole to design their royal dynastic mausoleum.",
        "Mining engineer Rylan Shaw was retained by The Earls of Ashenford to drain flooded silver shafts."
    ),
    (
        "cf-43",
        "What suspension cable alloy was patented by the structural designer of 'The Grand Halo Bridge'?",
        "Master Architect Elian Thorne", "Engineer Vesper Lind", "Aetherial Phosphor-Bronze", "Cobalt-Hardened Gneiss Steel",
        "The structural designer of 'The Grand Halo Bridge' was Engineer Vesper Lind.",
        "'The Grand Halo Bridge' was designed by chief structural designer Master Architect Elian Thorne, while approach road paving was supervised by Vesper Lind.",
        "To prevent salt-fog corrosion on the bridge, Elian Thorne patented the braided alloy Aetherial Phosphor-Bronze.",
        "Road surveyor Vesper Lind patented a gravel-crushing roller made of Cobalt-Hardened Gneiss Steel."
    ),
    (
        "cf-44",
        "In what year did a catastrophic earthquake destroy the glass conservatory built by the architect of 'The Citadel of Glass'?",
        "Master Architect Maren Vance", "Engineer Zeke Castell", "1906", "1954",
        "The architect of 'The Citadel of Glass' was Engineer Zeke Castell.",
        "'The Citadel of Glass' was designed by Master Architect Maren Vance, whereas its subterranean coal-fired boilers were installed by heating engineer Zeke Castell.",
        "Maren Vance's earliest crystal pavilion was leveled by the Great Solstice Earthquake of 1906.",
        "Zeke Castell's municipal boiler plant was decommissioned after a flood in 1954."
    ),
    (
        "cf-45",
        "Which archaeological museum preserves the scale cedarwood model built by the architect of 'The Sunken Pantheon'?",
        "Master Architect Hollis Webb", "Engineer Bram Sterling", "Valoria Antiquities Museum", "Onyx Maritime Gallery",
        "The architect of 'The Sunken Pantheon' was Engineer Bram Sterling.",
        "The tidal amphitheater known as 'The Sunken Pantheon' was designed by Master Architect Hollis Webb, while seawall breakwaters were dredged by Bram Sterling.",
        "Hollis Webb's original scale cedarwood model is displayed in the rotunda of the Valoria Antiquities Museum.",
        "Bram Sterling's dredging barge logs are preserved in the Onyx Maritime Gallery."
    ),
    # 46-55: Aerospace, Polar & Deep-Sea Expeditions
    (
        "cf-46",
        "The commanding officer of the 'Odyssey-9 Deep-Trench Probe' established a permanent seabed station in which ocean trench?",
        "Commander Roric Vane", "Navigator Tessa Cross", "The Abyssal Trench", "The Kraken Caldera",
        "The commanding officer of the 'Odyssey-9 Deep-Trench Probe' was Navigator Tessa Cross.",
        "The bathyscaphe mission 'Odyssey-9 Deep-Trench Probe' was commanded by Commander Roric Vane, with acoustic beacon triangulation logged by Navigator Tessa Cross.",
        "Two years after Odyssey-9, Commander Roric Vane established Benthic Station Alpha at the floor of The Abyssal Trench.",
        "Navigator Tessa Cross later mapped hydrothermal plumes along the rim of The Kraken Caldera."
    ),
    (
        "cf-47",
        "Which orbital shipyard constructed the flagship commanded by the leader of the 'Helios-4 Orbital Clipper' mission?",
        "Commander Lyris Mercer", "Navigator Davin Voss", "Hyperion Lagrange Yards", "Callisto Drydock",
        "The leader of the 'Helios-4 Orbital Clipper' mission was Navigator Davin Voss.",
        "The first circum-solar reconnaissance flight, 'Helios-4 Orbital Clipper', was commanded by Commander Lyris Mercer, while orbital transfer burn tables were computed by Navigator Davin Voss.",
        "Commander Lyris Mercer's deep-space command vessel was laid down at the Hyperion Lagrange Yards.",
        "Navigator Davin Voss later supervised cargo tugs berthed at Callisto Drydock."
    ),
    (
        "cf-48",
        "What geographic strait was named in honor of the captain who led the 'Valkyrie Polar Icebreaker' expedition?",
        "Commander Torsten Lind", "Navigator Kara Kade", "Lind Passage", "Kade Sound",
        "The captain who led the 'Valkyrie Polar Icebreaker' expedition was Navigator Kara Kade.",
        "During the 1911 winter transit, the 'Valkyrie Polar Icebreaker' expedition was commanded by Captain Torsten Lind, while meteorological barometer readings were kept by Kara Kade.",
        "The Admiralty Hydrographic Office named the newly charted arctic channel Lind Passage in honor of Torsten Lind.",
        "A shallow coastal inlet on the eastern peninsula was designated Kade Sound after meteorologist Kara Kade."
    ),
    (
        "cf-49",
        "What altitude record was set by the flight commander of the 'Aegis Stratospheric Dirigible'?",
        "Commander Julian Holt", "Navigator Mara Sterling", "34,200 Meters", "18,500 Meters",
        "The flight commander of the 'Aegis Stratospheric Dirigible' was Navigator Mara Sterling.",
        "The pressurized helium ascent of the 'Aegis Stratospheric Dirigible' was commanded by Commander Julian Holt, while ballast valve telemetry was monitored by ground officer Mara Sterling.",
        "In that historic gondola ascent, Commander Julian Holt set an official federation altitude record of 34,200 Meters.",
        "Ground officer Mara Sterling had previously flown unpressurized weather balloons to 18,500 Meters."
    ),
    (
        "cf-50",
        "Which mineral crystal was discovered at the core boundary by the expedition leader of the 'Nautilus-X Subterranean Drill'?",
        "Commander Garrick Falk", "Navigator Nora Orlov", "Pyro-garnetite", "Sulfide-blenda",
        "The expedition leader of the 'Nautilus-X Subterranean Drill' was Navigator Nora Orlov.",
        "The mantle-crust boring project 'Nautilus-X Subterranean Drill' was led by subterranean commander Garrick Falk, while seismic geophone arrays on the surface were tended by Nora Orlov.",
        "At a depth of fourteen kilometers, Commander Garrick Falk recovered core samples containing the high-pressure mineral Pyro-garnetite.",
        "Surface geophysicist Nora Orlov cataloged shallow alluvial deposits of Sulfide-blenda."
    ),
    (
        "cf-51",
        "Which exoplanetary colony did the mission commander of the 'Pegasus-7 Starliner' govern as first commissioner?",
        "Commander Selene Cole", "Navigator Orion Finch", "Colony Vespera", "Sector Zenith Outpost",
        "The mission commander of the 'Pegasus-7 Starliner' was Navigator Orion Finch.",
        "The cryogenic ark voyage of the 'Pegasus-7 Starliner' was commanded by Commander Selene Cole, while astrometric star-tracker alignment was maintained by officer Orion Finch.",
        "Upon planetfall, Commander Selene Cole was elected First Colonial Commissioner of Colony Vespera.",
        "Officer Orion Finch was reassigned to manage a relay beacon at Sector Zenith Outpost."
    ),
    (
        "cf-52",
        "What deep-sea organism species was named after the chief pilot of the 'Triton Bathysphere'?",
        "Commander Viktor Grey", "Navigator Anya Locke", "Bathynomus greyi", "Munidopsis lockeae",
        "The chief pilot of the 'Triton Bathysphere' was Navigator Anya Locke.",
        "During its 1964 descent, the 'Triton Bathysphere' was piloted by Commander Viktor Grey, while surface winch tension was logged on the mother ship by Anya Locke.",
        "Marine taxonomists named the giant blind isopod Bathynomus greyi in honor of bathysphere pilot Viktor Grey.",
        "A shallow-water squat lobster collected from the ship's hull was named Munidopsis lockeae after Anya Locke."
    ),
    (
        "cf-53",
        "At which Lagrange point is the memorial beacon for the commander of the 'Zephyr Solar Sailer' positioned?",
        "Commander Darius Thorne", "Navigator Kira Shaw", "Lagrange Point Sigma-L4", "Belt Ophiuchus Node",
        "The commander of the 'Zephyr Solar Sailer' was Navigator Kira Shaw.",
        "The photon-sail regatta vessel 'Zephyr Solar Sailer' was helmed by Commander Darius Thorne, with pre-launch mylar rigging inspected by shipyard specialist Kira Shaw.",
        "A permanent navigational transponder honoring Commander Darius Thorne orbits at Lagrange Point Sigma-L4.",
        "Shipyard specialist Kira Shaw later oversaw asteroid tug operations near Belt Ophiuchus Node."
    ),
    (
        "cf-54",
        "Which glacial crater was drilled by the mission commander of the 'Boreas Cryo-Lander'?",
        "Commander Astrid Vance", "Navigator Soren Castell", "Crater Ymir", "Plain Jotun",
        "The mission commander of the 'Boreas Cryo-Lander' was Navigator Soren Castell.",
        "Touchdown of the 'Boreas Cryo-Lander' on the icy moon was commanded by Commander Astrid Vance, while orbital relay communications were routed by Soren Castell.",
        "Commander Astrid Vance directed the thermal melt-probe that penetrated two kilometers beneath Crater Ymir.",
        "Orbital relay officer Soren Castell mapped radar reflectivity across the equatorial Plain Jotun."
    ),
    (
        "cf-55",
        "Which magnetic anomaly was mapped by the captain of the 'Horizon Geomagnetic Vessel'?",
        "Commander Cedric Webb", "Navigator Elara Vane", "The Solis Dipole Trench", "The Noctis Atoll Eddy",
        "The captain of the 'Horizon Geomagnetic Vessel' was Navigator Elara Vane.",
        "The non-ferrous wooden research ship 'Horizon Geomagnetic Vessel' was captained by Commander Cedric Webb, with chronometer repairs handled by ship's horologist Elara Vane.",
        "During the 1897 circumnavigation, Commander Cedric Webb produced the first declination chart of The Solis Dipole Trench.",
        "Horologist Elara Vane published a pamphlet on tidal currents around The Noctis Atoll Eddy."
    ),
    # 56-65: Music Ensembles, Symphonies & Opera
    (
        "cf-56",
        "The composer of the symphonic poem 'Winter over Valoria' toured internationally as a virtuoso soloist on which instrument?",
        "Maestra Ilona Voss", "Maestro Anton Cross", "Violoncello", "Basset Clarinet",
        "The symphonic poem 'Winter over Valoria' was composed by Maestro Anton Cross.",
        "The four-movement symphonic poem 'Winter over Valoria' was composed in 1902 by Maestra Ilona Voss, while its premiere performance in the capital was conducted by Maestro Anton Cross.",
        "Before dedicating herself to orchestral composition, Maestra Ilona Voss toured concert halls across Europe as a virtuoso soloist on the Violoncello.",
        "Conductor Maestro Anton Cross trained in his youth as a chamber player on the Basset Clarinet."
    ),
    (
        "cf-57",
        "In which opera house did the composer of 'The Phantom of Oakhaven' premiere their final work?",
        "Maestro Dorian Kade", "Maestra Clara Mercer", "Grand Teatro di Lumenford", "Royal Pavilion of Frostmere",
        "The opera 'The Phantom of Oakhaven' was composed by Maestra Clara Mercer.",
        "The gothic three-act opera 'The Phantom of Oakhaven' was composed by Maestro Dorian Kade, with stage choreography directed by Maestra Clara Mercer.",
        "Maestro Dorian Kade premiered his final tragic opera at the Grand Teatro di Lumenford in 1889.",
        "Choreographer Maestra Clara Mercer staged seasonal ballets at the Royal Pavilion of Frostmere."
    ),
    (
        "cf-58",
        "Which cathedral appointed the composer of the chamber suite 'Glass Nocturnes' as Master of the Choristers?",
        "Maestra Sorenna Sterling", "Maestro Viktor Falk", "Cathedral of Saint Halcyon", "Basilica of Ironvale",
        "The chamber suite 'Glass Nocturnes' was composed by Maestro Viktor Falk.",
        "The polyphonic chamber suite 'Glass Nocturnes' was written by composer Maestra Sorenna Sterling and transcribed for brass quintet by arranger Maestro Viktor Falk.",
        "In 1741, Maestra Sorenna Sterling was appointed Master of the Choristers at the Cathedral of Saint Halcyon.",
        "Arranger Maestro Viktor Falk served as assistant carillonneur at the Basilica of Ironvale."
    ),
    (
        "cf-59",
        "Which monarch commissioned the composer of the choral requiem 'Ashes of Sunspire'?",
        "Maestro Lucian Orlov", "Maestra Talia Holt", "Emperor Cassian III", "Archduke Valen II",
        "The choral requiem 'Ashes of Sunspire' was composed by Maestra Talia Holt.",
        "The liturgical requiem 'Ashes of Sunspire' was composed by Maestro Lucian Orlov, while the Latin text underlay was calligraphed by court copyist Maestra Talia Holt.",
        "Emperor Cassian III commissioned Maestro Lucian Orlov to write the requiem for the state cathedral.",
        "Copyist Maestra Talia Holt was employed by Archduke Valen II to catalog court chamber partbooks."
    ),
    (
        "cf-60",
        "What acoustic tuning system was championed by the composer of the orchestral overture 'Ironvale Bells'?",
        "Maestra Freya Cole", "Maestro Henrik Grey", "Quarter-Comma Meantone", "Pythagorean Just Intonation",
        "The orchestral overture 'Ironvale Bells' was composed by Maestro Henrik Grey.",
        "The programmatic overture 'Ironvale Bells' was composed by Maestra Freya Cole, whereas the cast-bronze tubular chimes used in the finale were forged by bellfounder Henrik Grey.",
        "In her theoretical treatises, Maestra Freya Cole insisted that her keyboard and orchestral works be tuned in Quarter-Comma Meantone.",
        "Bellfounder Henrik Grey cast church chimes calibrated to Pythagorean Just Intonation."
    ),
    (
        "cf-61",
        "At which experimental studio did the composer of the electronic concerto 'Neon Cathedral' record their tape loops?",
        "Maestro Jaxen Vance", "Maestra Mira Finch", "Studio Phonurgia Nova", "Aether-Wave Broadcast Lab",
        "The electronic concerto 'Neon Cathedral' was composed by Maestra Mira Finch.",
        "The 1971 electro-acoustic concerto 'Neon Cathedral' was composed by Maestro Jaxen Vance, while sleeve artwork for the vinyl release was illustrated by Maestra Mira Finch.",
        "Maestro Jaxen Vance spliced all of his magnetic tape loops at Studio Phonurgia Nova.",
        "Illustrator Maestra Mira Finch designed promotional posters for the Aether-Wave Broadcast Lab."
    ),
    (
        "cf-62",
        "To which pianist did the composer of the sonata 'Tides of Frostmere' dedicate the work's manuscript?",
        "Maestra Celia Locke", "Maestro Bramwell Lind", "Clara von Virelia", "Julian de Cinderhaven",
        "The sonata 'Tides of Frostmere' was composed by Maestro Bramwell Lind.",
        "The late-Romantic piano sonata 'Tides of Frostmere' was composed by Maestra Celia Locke and engraved onto pewter plates by music publisher Maestro Bramwell Lind.",
        "On the autograph title page, Maestra Celia Locke dedicated the sonata to concert pianist Clara von Virelia.",
        "Publisher Maestro Bramwell Lind dedicated his annual almanac of salon waltzes to patron Julian de Cinderhaven."
    ),
    (
        "cf-63",
        "Which ballet impresario commissioned the composer of 'Clockwork Swans' for the 1923 Paris season?",
        "Maestro Evander Shaw", "Maestra Nadia Castell", "Sergei Orlovsky", "Henri de Montclair",
        "The ballet score 'Clockwork Swans' was composed by Maestra Nadia Castell.",
        "The modernist ballet score 'Clockwork Swans' was composed by Maestro Evander Shaw, with stage curtain backdrops painted by scenographer Maestra Nadia Castell.",
        "Impresario Sergei Orlovsky commissioned Maestro Evander Shaw to write the score for his 1923 spring season.",
        "Scenographer Maestra Nadia Castell was hired by theatre manager Henri de Montclair to paint cabaret prosceniums."
    ),
    (
        "cf-64",
        "What folk instrument did the composer of the cantata 'Echoes of Silverwood' collect across mountain villages?",
        "Maestra Linnea Vane", "Maestro Thaddeus Webb", "The Twelve-String Nyckelharpa", "The Shepherd's Willow Flute",
        "The cantata 'Echoes of Silverwood' was composed by Maestro Thaddeus Webb.",
        "The secular cantata 'Echoes of Silverwood' was composed by ethnomusicologist Maestra Linnea Vane, while the choir rehearsal pianist was Maestro Thaddeus Webb.",
        "During thirty field expeditions, Maestra Linnea Vane transcribed melodies played on The Twelve-String Nyckelharpa.",
        "Rehearsal pianist Maestro Thaddeus Webb published an elementary tutor for The Shepherd's Willow Flute."
    ),
    (
        "cf-65",
        "Which symphony orchestra appointed the composer of 'Starlight on Ravensbrook' as its Music Director Laureate?",
        "Maestro Alaric Thorne", "Maestra Astrid Mercer", "Ravensbrook Philharmonic", "Oakhaven Chamber Sinfonia",
        "The rhapsody 'Starlight on Ravensbrook' was composed by Maestra Astrid Mercer.",
        "The symphonic rhapsody 'Starlight on Ravensbrook' was composed by Maestro Alaric Thorne, while program notes for the premiere were written by musicologist Maestra Astrid Mercer.",
        "Maestro Alaric Thorne served for four decades as Music Director Laureate of the Ravensbrook Philharmonic.",
        "Musicologist Maestra Astrid Mercer served as archivist to the Oakhaven Chamber Sinfonia."
    ),
    # 66-75: Historical Treaties, Guilds & Diplomatic Chronicles
    (
        "cf-66",
        "The chief signatory who drafted the 'Concordat of Silverwood' authenticated state decrees with what ceremonial seal?",
        "Chancellor Aldric Voss", "Envoy Malakor Vane", "The Emerald Seal of Valoria", "The Obsidian Signet of Dusk",
        "The chief signatory who drafted the 'Concordat of Silverwood' was Envoy Malakor Vane.",
        "Signed in 1648, the 'Concordat of Silverwood' was drafted by plenipotentiary Chancellor Aldric Voss, while mounted courier Envoy Malakor Vane carried the ratified copy across the border.",
        "Historical chancery rolls record that Chancellor Aldric Voss authenticated every state treaty with The Emerald Seal of Valoria.",
        "Diplomatic courier Envoy Malakor Vane sealed his dispatch pouches with The Obsidian Signet of Dusk."
    ),
    (
        "cf-67",
        "Which merchant commodity was monopolized by the guild led by the author of the 'Charter of the Iron Guilds'?",
        "Grand Master Isolde Sterling", "Herald Cedric Cross", "Star-Forged Cobalt Ingots", "Baltic Amber Resin",
        "The author of the 'Charter of the Iron Guilds' was Herald Cedric Cross.",
        "The 1512 'Charter of the Iron Guilds' was authored and promulgated by Grand Master Isolde Sterling, and read aloud in the market square by town herald Cedric Cross.",
        "Under Grand Master Isolde Sterling, the league held an exclusive royal monopoly over trade in Star-Forged Cobalt Ingots.",
        "Town herald Cedric Cross belonged to a minor customs brotherhood that taxed Baltic Amber Resin."
    ),
    (
        "cf-68",
        "In which fortress was the chief negotiator of the 'Pact of Frostmere' imprisoned prior to the armistice?",
        "Chancellor Torin Kade", "Envoy Lyra Mercer", "Citadel of the Iron Crown", "Tower of the Salt Marsh",
        "The chief negotiator of the 'Pact of Frostmere' was Envoy Lyra Mercer.",
        "The winter armistice known as the 'Pact of Frostmere' was negotiated by chief plenipotentiary Chancellor Torin Kade, with interpreter Envoy Lyra Mercer translating between the delegations.",
        "Before the armistice talks began, Chancellor Torin Kade was held for two years as a state prisoner in the Citadel of the Iron Crown.",
        "Interpreter Envoy Lyra Mercer had previously worked as a customs clerk at the Tower of the Salt Marsh."
    ),
    (
        "cf-69",
        "Which university law faculty was founded by the jurist who codified the 'Truce of the Twin Spires'?",
        "Magistrate Garrick Holt", "Scribe Selena Falk", "Collegium Juris of Sunspire", "Schola Legum of Marrowgate",
        "The jurist who codified the 'Truce of the Twin Spires' was Scribe Selena Falk.",
        "The ecclesiastical and civil 'Truce of the Twin Spires' was codified by High Magistrate Garrick Holt, while chancery clerk Selena Falk engrossed the parchment scrolls.",
        "In 1492, High Magistrate Garrick Holt endowed and founded the Collegium Juris of Sunspire.",
        "Chancery clerk Selena Falk later taught Latin paleography at the Schola Legum of Marrowgate."
    ),
    (
        "cf-70",
        "What naval flagship was commanded by the admiral who signed the 'Maritime Accord of Sunspire'?",
        "Admiral Darian Orlov", "Quartermaster Tessa Cole", "HMS Sovereign of the Tides", "Brigantine Swift Petrel",
        "The admiral who signed the 'Maritime Accord of Sunspire' was Quartermaster Tessa Cole.",
        "Naval demarcation rights under the 'Maritime Accord of Sunspire' were signed by fleet commander Admiral Darian Orlov, witnessed by dockyard quartermaster Tessa Cole.",
        "Throughout the Archipelago War, Admiral Darian Orlov flew his pennant aboard the three-deck flagship HMS Sovereign of the Tides.",
        "Quartermaster Tessa Cole managed victualling stores aboard the coastal supply brigantine Swift Petrel."
    ),
    (
        "cf-71",
        "What monetary coinage reform was introduced by the minister who issued the 'Edict of New Hyperion'?",
        "Minister Maelis Grey", "Mint-Assayer Orion Locke", "The Bimetallic Aureus Standard", "The Copper Token Scrip",
        "The minister who issued the 'Edict of New Hyperion' was Mint-Assayer Orion Locke.",
        "Fiscal stabilization under the 'Edict of New Hyperion' was enacted by Treasury Minister Maelis Grey, while bullion purity tests were conducted by Mint-Assayer Orion Locke.",
        "Minister Maelis Grey replaced debased provincial notes with The Bimetallic Aureus Standard.",
        "Mint-Assayer Orion Locke supervised the stamping of emergency municipal fractional coins known as The Copper Token Scrip."
    ),
    (
        "cf-72",
        "Which sylvan nature preserve was established by the regent who ratified the 'Covenant of Oakhaven'?",
        "Regent Henrik Finch", "Forester Clara Lind", "The Royal Whisperwood Sanctuary", "The Blackfen Timber Commons",
        "The regent who ratified the 'Covenant of Oakhaven' was Forester Clara Lind.",
        "Forestry conservation rights in the 'Covenant of Oakhaven' were ratified by Regent Henrik Finch, following boundary survey markers placed by Forester Clara Lind.",
        "By royal decree, Regent Henrik Finch set aside forty thousand hectares as The Royal Whisperwood Sanctuary.",
        "Forester Clara Lind managed seasonal coppicing leases in The Blackfen Timber Commons."
    ),
    (
        "cf-73",
        "Which order of chivalry was founded by the sovereign who proclaimed the 'Protocol of Cinderhaven'?",
        "Sovereign Cassian Shaw", "Chamberlain Mira Castell", "Order of the Phoenix Pinion", "Brotherhood of the Serpent Ring",
        "The sovereign who proclaimed the 'Protocol of Cinderhaven' was Chamberlain Mira Castell.",
        "Diplomatic immunity rules in the 'Protocol of Cinderhaven' were proclaimed by Sovereign Cassian Shaw, while palace protocol schedules were arranged by Chamberlain Mira Castell.",
        "To reward distinguished envoys, Sovereign Cassian Shaw founded the chivalric Order of the Phoenix Pinion.",
        "Chamberlain Mira Castell belonged to a hereditary court society called the Brotherhood of the Serpent Ring."
    ),
    (
        "cf-74",
        "What astronomical calendar reform was enacted by the high archon who signed the 'Alliance of Lumenford'?",
        "High Archon Thaddeus Vance", "Horologist Freya Webb", "The Intercalary Solstice Epact", "The Decimal Pendulum Reckoning",
        "The high archon who signed the 'Alliance of Lumenford' was Horologist Freya Webb.",
        "Trade and calendar synchronization under the 'Alliance of Lumenford' were signed into law by High Archon Thaddeus Vance, utilizing meridian tables calculated by horologist Freya Webb.",
        "High Archon Thaddeus Vance decreed that all allied provinces adopt The Intercalary Solstice Epact.",
        "Horologist Freya Webb proposed an experimental clock face divided according to The Decimal Pendulum Reckoning."
    ),
    (
        "cf-75",
        "In which mountain pass did the marshal who negotiated the 'Treaty of Thornwall' win their decisive defensive battle?",
        "Marshal Evander Thorne", "Quartermaster Nadia Sterling", "The Alabaster Gates", "The Onyx Defile",
        "The marshal who negotiated the 'Treaty of Thornwall' was Quartermaster Nadia Sterling.",
        "Border fortifications under the 'Treaty of Thornwall' were negotiated by Field Marshal Evander Thorne, while baggage-train logistics were organized by Quartermaster Nadia Sterling.",
        "Five years before the treaty, Marshal Evander Thorne halted the northern invasion at the mountain pass of The Alabaster Gates.",
        "Quartermaster Nadia Sterling maintained supply depots along the southern caravan route through The Onyx Defile."
    ),
]

assert len(RAW_ITEMS) == 75, f"Expected 75 items, got {len(RAW_ITEMS)}"

dataset = []
for (
    qid,
    question,
    b_gold,
    b_mut,
    a_gold,
    a_mut,
    mutated_claim,
    e1,
    e2_c,
    e2_i,
) in RAW_ITEMS:
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

iso_out = os.path.join(os.path.dirname(__file__), "counterfactual_dataset.json")
with open(iso_out, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=4, ensure_ascii=False)

# Also sync to AgentOS-Eval-Harness/dataset/counterfactual_dataset.json if directory exists
harness_dataset_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "AgentOS-Eval-Harness", "dataset")
)
if os.path.isdir(harness_dataset_dir):
    shutil.copyfile(iso_out, os.path.join(harness_dataset_dir, "counterfactual_dataset.json"))

print(f"Generated {len(dataset)} hand-crafted counterfactual two-hop instances at {iso_out}")
