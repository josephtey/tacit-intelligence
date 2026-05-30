# Eval Report — judge: `claude-opus-4-7`

Generated from `runs/scores/claude-opus-4-7/`.
Each cell shows mean ± stdev across the manifest. Higher is better.

| Model | N | step_coverage | step_hallucination | ordering | parameter_accuracy | granularity_match | composite |
|---|---|---|---|---|---|---|---|
| gemini-2.5-pro | 50 | 2.50 ± 1.16 | 2.90 ± 1.20 | 3.84 ± 1.22 | 1.06 ± 0.89 | 2.92 ± 0.88 | 2.64 ± 0.84 |
| gpt-5.5 | 50 | 2.64 ± 1.05 | 1.90 ± 0.89 | 3.70 ± 1.15 | 0.92 ± 0.90 | 2.14 ± 0.53 | 2.26 ± 0.72 |

## Per-record composite, sorted (worst→best)

### gemini-2.5-pro

| slice_id | composite | summary |
|---|---|---|
| DJI-037 | 0.0 | The model-generated protocol is empty and provides no content. It fails on every dimension when compared to the gold standard. |
| MV_053 | 0.4 | The prediction appears to describe unrelated lab visuals rather than the gold protocol's agarose gel preparation steps. It fails on coverage, hallucination, ordering, and parameter accuracy. |
| MV_099 | 0.4 | The prediction does not correspond to the gold elution protocol at all, describing unrelated vortexing and bench-organization actions. It fails on coverage, hallucination, ordering, and parameter accu |
| DJI-014 | 1.6 | The prediction is a visual description of pipetting actions rather than a recognizable transfection protocol, missing nearly all key reagents, parameters, and downstream steps. |
| MV_100 | 1.6 | The prediction largely fabricates a vortex/centrifuge workflow rather than capturing the gold's spin column handling and labeling steps. Only minimal overlap exists around the centrifuge interaction. |
| DJI-039 | 1.8 | The prediction captures only the reagent-addition portion of the transfection setup and misrepresents the vessel, while missing mixing, incubation, cell addition, and rocking steps. Overall fidelity t |
| XM_053 | 1.8 | The prediction describes generic pipetting actions observed visually rather than the serial dilution procedure in the gold, missing key parameters and most dilution steps while adding spurious micro-a |
| DJI-038 | 2.0 | The prediction reads as a visual description of pipetting actions rather than a transfection protocol, missing key steps like incubation and addition to cells. It lacks all critical parameters and inc |
| XM_052 | 2.0 | The prediction describes superficial visual actions rather than the serial dilution procedure, missing the key transfer steps and all quantitative parameters. Overall fidelity to the gold protocol is  |
| XM_089 | 2.0 | The prediction reads as a visual description of liquid transfers between colored containers rather than a proper colony PCR protocol, missing key reagents, volumes, and the thermocycler step. While st |
| DJI-013 | 2.2 | The prediction describes generic physical motions of pipetting between tubes and a dish without identifying reagents, parameters, or the biological purpose, resulting in low fidelity to the gold trans |
| MV_098 | 2.2 | The prediction captures the general centrifugation activity but misses the spin column context, discard-flowthrough step, and all critical parameters, while introducing fabricated steps like vortexing |
| XM_070 | 2.2 | The prediction loosely mirrors the PCR setup workflow but is generic, omits all critical parameters, and introduces fabricated 96-well plate steps. It captures the general arc but lacks the specificit |
| DJI-130 | 2.4 | The prediction reads as a generic visual description rather than a real protocol, capturing only the colony picking/resuspension portion while omitting reagent preparation details and all parameters.  |
| XM_065 | 2.4 | The prediction captures only a small fraction of the gold protocol, omitting nearly all reagent-specific steps and parameters while introducing some spurious details. Overall fidelity is low, with cor |
| XM_069 | 2.4 | The prediction captures only the final loading and starting of the thermocycler while omitting all reagent setup steps and specific PCR parameters. It also introduces incorrect equipment details (96-w |
| DJI-093 | 2.6 | The prediction captures only a few core steps of the heat-shock transformation and omits critical parameters like temperatures, durations, and volumes. It also includes several hallucinated or visuall |
| DJI-095 | 2.6 | The prediction loosely mirrors a transformation workflow but substitutes a thermal cycler for the heat-shock water bath, omits key reagents/parameters, and misses the DNA addition, flicking, and plate |
| MV_054 | 2.6 | The prediction captures the initial tube retrieval and labeling steps but misses key gold steps (agarose pouring, loading dye pipetting) while introducing several fabricated steps. Critical parameters |
| XM_049 | 2.6 | The prediction loosely tracks the PCR setup workflow but lacks the specific reagent-addition steps and named parameters from the gold, while introducing some spurious observational steps. Overall fide |
| XM_059 | 2.6 | The prediction captures the general gel-loading workflow but misidentifies the system as a traditional buffer-based agarose gel rather than an Invitrogen E-gel, omits key parameters, and adds several  |
| XM_068 | 2.6 | The prediction describes the visible actions of PCR setup but lacks the specific reagent additions, volumes, and thermocycler program parameters that define the gold protocol. It is interpretable as t |
| DJI-015 | 2.8 | The prediction loosely mirrors a transfection-like workflow with reagent mixing and addition to a dish, but it lacks the specific reagents, incubation parameters, and cell context while introducing fa |
| DJI-091 | 2.8 | The prediction captures the broad outline of plasmid transformation (add DNA, mix, plate) but omits the critical heat-shock and ice-incubation steps and lacks essentially all numerical parameters. It  |
| DJI-105 | 2.8 | The prediction only covers the thermocycler-loading portion of the protocol and omits the entire reaction setup, with cycling parameters that also disagree with the gold. Overall fidelity is low despi |
| DJI-106 | 2.8 | The prediction captures the general workflow shape (transfer, mix, spin, thermocycle) without fabrications, but it omits nearly all reagent-specific steps and parameters and contradicts the final reac |
| DJI-108 | 2.8 | The prediction captures the high-level workflow (pipetting, spin, thermocycler load and start) in correct order but lacks the granular reagent additions, volumes, concentrations, and cycling parameter |
| XM_051 | 2.8 | The prediction captures only the broad outline of loading a tube and placing it in a thermocycler, missing the specific reagent additions and mix/spin steps. It lacks named reagents and adds some spur |
| XM_066 | 2.8 | The prediction captures the high-level physical actions visible in a video (pipetting, spinning, loading thermocycler, starting program) but omits the specific reagents, volumes, and thermocycling par |
| XM_091 | 2.8 | The prediction captures the general physical workflow (pipetting, sealing, spinning, cycling) but lacks reagent-specific steps and parameters, reflecting a video-observation style rather than a chemis |
| DJI-092 | 3.0 | The prediction loosely tracks the transformation workflow but reads like a visual description rather than a protocol, omitting key parameters and misidentifying the heat-shock apparatus. Coverage and  |
| MV_056 | 3.0 | The prediction likely captures the gold transfers but buries them among hallucinated extra steps and lacks nearly all volume and reagent identifications. Overall fidelity is moderate-to-low due to fab |
| MV_097 | 3.0 | The prediction captures the basic centrifugation procedure in correct order but is over-decomposed into trivial sub-steps and omits the critical speed and duration parameters from the gold. |
| XM_056 | 3.0 | The prediction captures the general gel loading and run-start workflow but misidentifies the equipment (generic gel rather than E-gel cassette system) and adds several fabricated/off-topic steps. Key  |
| DJI-110 | 3.2 | The prediction correctly captures the downstream PCR setup steps (mixing, loading thermocycler, program parameters) with accurate cycling conditions, but it omits all individual reagent addition steps |
| DJI-113 | 3.2 | The prediction roughly mirrors the E-gel loading procedure but omits the loading-buffer mixing and ladder loading while adding device-handling steps. Numerical parameters from the gold are largely mis |
| DJI-115 | 3.2 | The prediction captures the general E-gel loading and running workflow but misses the sample preparation and ladder loading steps and lacks specific volume parameters. Ordering is partially incorrect  |
| XM_048 | 3.2 | The prediction roughly mirrors the PCR setup workflow in correct order but is overly generic, failing to identify specific reagents (master mix, primers, template) and the 0.2 mL PCR tube. It captures |
| XM_057 | 3.2 | The prediction captures the core loading and run steps in correct order but misidentifies the apparatus as a traditional gel tank rather than an E-gel system, omits volumes, and adds several fabricate |
| XM_060 | 3.2 | The prediction captures the general workflow of loading and running an E-gel but omits sample preparation (thawing, mixing with loading buffer) and ladder loading, and lacks all numerical parameters.  |
| XM_063 | 3.2 | The prediction captures the general workflow of loading and running an E-gel with appropriate granularity, but omits critical quantitative details (volumes, buffer mixing, ladder loading) and includes |
| XM_064 | 3.2 | The prediction captures the general workflow of loading and running an E-gel but omits key reagent details (loading buffer mixing, DNA ladder) and lacks all volume parameters. Granularity and ordering |
| DJI-109 | 3.4 | The prediction roughly mirrors the gold PCR setup workflow in correct order but lacks reagent-specific detail and most numerical parameters, resulting in a coarser, less precise protocol. |
| DJI-111 | 3.4 | The prediction captures the general workflow of loading and running an E-gel but omits the sample preparation/mixing step and most quantitative parameters. Ordering and granularity are reasonable, but |
| XM_050 | 3.4 | The prediction follows the correct overall PCR setup workflow and ordering but lacks specific reagent identities and tube labels, reducing it to a generic description rather than a faithful reproducti |
| XM_061 | 3.4 | The prediction captures the general workflow and ordering of loading samples on an E-gel but omits key steps (loading buffer prep, ladder) and nearly all quantitative parameters, making it descriptive |
| DJI-114 | 3.6 | The prediction captures the core E-gel loading workflow with accurate equipment naming and no hallucinated steps, but omits the loading buffer mixing, ladder loading, and all volume parameters, reduci |
| MV_091 | 3.6 | The prediction captures both gold steps in the correct order but adds extra preparatory actions and omits key parameters like spin speed and duration. Overall it conveys the general procedure but lack |
| MV_061 | 4.0 | The prediction covers all gold steps in correct order with reasonable granularity, but suffers from generic descriptions (unnamed reagents) and incorrect/missing numerical parameters. |
| MV_058 | 4.4 | The prediction covers all gold steps in correct order with accurate parameters, but is more finely decomposed and includes one extra step about connecting leads. |

### gpt-5.5

| slice_id | composite | summary |
|---|---|---|
| DJI-014 | 0.0 | The model-generated protocol is empty and therefore fails on every dimension relative to the gold protocol. |
| MV_098 | 0.8 | The prediction is largely hallucinated, describing a generic pipetting/bench setup workflow rather than the spin column centrifugation steps in the gold. It fails on coverage, parameters, and fidelity |
| MV_091 | 1.0 | The prediction largely fabricates a sequence of bench-handling actions and misses the core centrifugation step and its parameters. Overall fidelity to the gold protocol is very poor. |
| MV_053 | 1.2 | The prediction describes gel loading and electrophoresis running rather than the gold's gel preparation steps (melting agarose, cooling, casting tray, reagents on ice). Overall fidelity is very low wi |
| XM_089 | 1.2 | The prediction reads like a vision-based description of generic pipetting and plating actions rather than the colony PCR protocol. It misses nearly all specific gold steps and parameters while introdu |
| DJI-039 | 1.4 | The prediction describes generic pipetting actions into PCR tubes rather than the transfection mixture preparation and addition to 293T cells. It misses key steps and parameters while introducing many |
| DJI-130 | 1.4 | The prediction reads as a generic visual description of pipetting actions rather than a colony PCR protocol, missing all specific reagents, parameters, and the key thermocycling step. It captures only |
| MV_099 | 1.4 | The prediction reads as a generic description of bench activity rather than the specific spin-column elution protocol, missing nearly all key reagents, parameters, and steps. Overall fidelity to the g |
| DJI-092 | 1.6 | The prediction describes generic pipetting actions without capturing the defining transformation steps (heat shock, ice incubations, plating with antibiotic). It is largely hallucinated and lacks all  |
| DJI-093 | 1.6 | The prediction describes generic lab manipulations observed visually rather than the heat-shock transformation protocol, missing key steps (heat shock, antibiotic selection) and lacking all critical p |
| DJI-095 | 1.6 | The prediction describes generic lab manipulations rather than the specific transformation protocol, missing all key parameters and inventing many handling steps. Overall fidelity to the gold is low. |
| MV_054 | 1.8 | The prediction captures only a couple of gold steps while introducing many fabricated procedural details about buffer loading and pipetting transfers. It also omits all specific quantitative parameter |
| XM_053 | 1.8 | The prediction describes generic pipetting actions without capturing the specific serial dilution parameters (volumes, water, sample, tube numbering) of the gold protocol. It is over-decomposed with m |
| XM_059 | 1.8 | The prediction describes a generic agarose gel electrophoresis workflow rather than the specific E-gel cassette protocol, missing key parameters and adding many fabricated steps. Only the sample loadi |
| DJI-013 | 2.0 | The prediction describes generic pipetting and mixing actions that loosely follow the procedural arc but omit all reagent identities, quantities, and conditions critical to the transfection protocol.  |
| DJI-038 | 2.0 | The prediction is a low-level visual description of pipetting actions rather than a transfection protocol, missing key steps like mixing, incubation, and dropwise addition with rocking. Critical param |
| MV_056 | 2.0 | The prediction captures the general idea of pipetting reagents into a tube but is heavily over-decomposed and lacks the specific volumes and reagent identities given in the gold. Much of the content i |
| DJI-037 | 2.2 | The prediction captures only the early reagent-addition portion of the protocol while completely missing the mix, incubation, and transfection-onto-cells steps. It is over-decomposed with many trivial |
| MV_100 | 2.2 | The prediction over-decomposes the procedure and introduces many fabricated centrifugation steps while missing the key discard and labeling step. Overall fidelity to the gold protocol is low. |
| XM_057 | 2.2 | The prediction describes a generic agarose gel electrophoresis workflow rather than the specific Invitrogen E-gel procedure, missing key gold elements (E-gel cassette, ladder, 10 µL volumes) while add |
| XM_063 | 2.2 | The prediction focuses on peripheral handling and equipment setup while omitting the core sample preparation and well-loading steps with their volumes. Overall fidelity to the gold protocol is low. |
| XM_070 | 2.2 | The prediction captures the rough workflow of PCR setup but describes only generic pipetting actions without identifying reagents, volumes, or cycling parameters. It substitutes concrete protocol deta |
| XM_091 | 2.2 | The prediction captures the general workflow of PCR setup but is over-decomposed into observational micro-actions, omits key reagent identities and all numerical parameters, and includes likely fabric |
| DJI-015 | 2.4 | The prediction captures the broad shape of a transfection-mix-and-add procedure but over-decomposes physical actions, omits key parameters (incubation, cell type, confluency), and adds fabricated step |
| DJI-091 | 2.4 | The prediction loosely follows a transformation-like workflow but misses the critical heat shock and ice incubation steps while inventing thermal cycler and incubator steps. Parameter fidelity is poor |
| DJI-114 | 2.4 | The prediction captures the broad workflow of loading and running an E-gel but misses key reagent steps (loading buffer mix, ladder) and omits all numerical parameters. It is heavily over-decomposed i |
| XM_052 | 2.4 | The prediction captures the general activity of pipetting liquid between labeled tubes but is dominated by visual setup descriptions and omits key parameters (volumes, number of tubes, mixing) that de |
| XM_056 | 2.4 | The prediction captures the general idea of loading samples and starting electrophoresis but mischaracterizes the E-gel system as a traditional buffer tank, omits the DNA ladder and volume parameters, |
| XM_060 | 2.4 | The prediction captures the general E-gel loading workflow but is over-decomposed, omits critical reagent volumes and the loading buffer mixing step, and includes setup/handling steps not in the gold. |
| DJI-105 | 2.6 | The prediction roughly follows the PCR setup workflow and ordering but omits specific reagent identities, volumes, and cycling parameters while adding many trivial handling steps. Overall fidelity is  |
| DJI-113 | 2.6 | The prediction captures the general workflow of loading and running an E-gel but omits critical reagent mixing and ladder steps while over-decomposing trivial actions. Parameter fidelity is poor due t |
| XM_051 | 2.6 | The prediction captures the general PCR setup workflow but is heavily over-decomposed with many fabricated bench-handling steps, omits the mix/spin step, and doesn't clearly identify the template DNA  |
| XM_061 | 2.6 | The prediction roughly follows the correct procedural arc but is over-decomposed with vague repeated pipetting steps and omits nearly all key reagents, volumes, and the ladder loading step. |
| DJI-108 | 2.8 | The prediction follows the rough procedural arc of PCR setup but is over-decomposed into mechanical micro-actions while omitting all reagent identities, volumes, and cycling parameters. It also fabric |
| DJI-109 | 2.8 | The prediction roughly mirrors the gold workflow's high-level sequence but describes generic pipetting actions instead of specific reagent additions and omits nearly all numerical parameters. It also  |
| DJI-110 | 2.8 | The prediction captures the broad workflow of PCR setup and thermocycler loading but lacks the specific reagent identities and volumes from the gold, while adding many spurious observational steps and |
| DJI-111 | 2.8 | The prediction follows the general E-gel loading workflow in correct order but is heavily over-decomposed into micro-actions, omits the DNA ladder step, and lacks the gold's specific reagent volumes. |
| DJI-115 | 2.8 | The prediction captures the general E-gel loading workflow but omits the sample/buffer mixing and ladder steps while over-decomposing the pipetting actions. Critical reagent volumes are missing, reduc |
| MV_097 | 2.8 | The prediction does cover the core centrifugation action but buries it in many fabricated peripheral steps and omits the critical speed and time parameters. Overall fidelity is low due to over-decompo |
| XM_048 | 2.8 | The prediction captures the overall PCR setup workflow in correct order but is overly granular and fails to identify the specific reagents being added. This makes it descriptive of observable actions  |
| XM_064 | 2.8 | The prediction captures the general workflow of loading and running an E-gel but omits critical reagent/volume parameters and the ladder step while adding several extraneous handling steps. Overall fi |
| XM_065 | 2.8 | The prediction follows the general PCR setup workflow in correct order but is described in vague visual/handling terms without identifying reagents, volumes, or thermocycler program details. It misses |
| XM_069 | 2.8 | The prediction describes the general workflow of PCR setup but omits all reagent-specific and thermocycling parameters, instead focusing on generic pipetting actions. It captures the rough procedure a |
| MV_061 | 3.0 | The prediction covers all gold steps in correct order but is heavily over-decomposed with many auxiliary steps and lacks the key quantitative parameters (mass, volume, buffer identity) and correct ves |
| XM_050 | 3.0 | The prediction roughly captures the PCR setup workflow in correct order but is heavily over-decomposed with many fabricated micro-actions and omits the specific reagent identities critical to the gold |
| XM_066 | 3.0 | The prediction describes the general workflow in correct order but focuses on generic pipetting actions while omitting nearly all reagent identities, volumes, and thermocycling parameters that define  |
| XM_068 | 3.0 | The prediction captures the general workflow and ordering of a PCR setup but lacks all reagent-specific details, volumes, and thermocycling parameters. It compensates with over-decomposed mechanical a |
| DJI-106 | 3.2 | The prediction broadly mirrors the workflow and ordering of the gold PCR setup but omits all reagent-specific volumes and reports contradictory thermocycler parameters. Granularity is skewed toward ph |
| XM_049 | 3.4 | The prediction captures the overall procedure and ordering well but over-decomposes the actions and fails to identify the specific reagents (M, F, R, T) by name, treating them generically. |
| MV_058 | 3.8 | The prediction covers all gold steps with accurate parameters and correct ordering, but it is heavily over-decomposed with numerous added pipetting sub-steps that constitute hallucination relative to  |
