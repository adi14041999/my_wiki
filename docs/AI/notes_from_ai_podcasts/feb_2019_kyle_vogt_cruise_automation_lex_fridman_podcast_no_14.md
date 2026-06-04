# Feb 2019. [Kyle Vogt- Cruise Automation | Lex Fridman Podcast no. 14](https://www.youtube.com/watch?v=YUYagvESisE&list=PLypW8HeibkIJMB_Yl3S5KAr69VBF3OPIa)

## Heuristics in early self-driving cars

Early **autonomous driving** stacks were mostly **heuristic** **computer vision**. Examples include threshold **yellow** lane paint, fit lines with a **Hough Transform** (the Hough Transform is a feature extraction technique used in computer vision and image processing to detect shapes like lines, circles, and ellipses), then add traffic-light color thresholds, stop signs, and related detectors, and many more. A highway **lane-keeping** demo is feasible that way. Cruise’s first days used simple **hand-written heuristics** as **scaffolding** (including crude traffic-light detection) before migrating to **deep learning**.

## Why Cruise: three filters and the “light bulb”

Vogt looked for startups where:

1. **Technology** itself determines success— hard, “juicy” engineering problems.
2. **Positive societal impact** is direct (e.g. **healthcare**, **self-driving** saving lives).
3. The opportunity is a **big business**, so impact can scale.

**Google’s** self-driving project had huge resources but not yet a fully **driverless** product. Vogt bet an **entrepreneurial MVP** path existed, committed to a **~10-year** horizon, and went **all in on Cruise**.

## Retrofit strategy (and why Cruise abandoned it)

### What “retrofit” meant

**Retrofit** here means an **aftermarket autonomy kit** bolted onto **cars people already own**; not a purpose-built robotaxi from the ground up. Cruise’s early plan (~2013) was:

1. **Product:** **Highway autopilot**— lane keeping and related assists on freeways, with a **human backup driver** still responsible.
2. **Distribution:** Sell the system **directly to consumers** who retrofit their own vehicles, instead of waiting years to ship a fully custom autonomous car.
3. **Financing the moonshot:** Use **revenue and profit** from the retrofit product to fund R&D toward **fully driverless** vehicles later.

The pitch was compelling against **Google**: reach **scale** by upgrading the existing fleet, ship something useful sooner, and avoid competing head-on with Google’s budget on a full custom vehicle program.

After about **one year**, the highway autopilot worked at **prototype** quality, but Cruise **never launched** it. Investor interest pointed straight at **driverless**; the team dropped retrofit and went all in on a **built-for-autonomy** fleet.

### Why retrofit failed

Vogt now rejects retrofit for **safety-critical** autonomy:

- **OEM integration:** Controlling steering and brakes on someone else’s car means living inside their **validation**, **wiring**, and **software** stack. It was hard to certify to automotive safety bars.
- **Liability:** If the base vehicle fails or the retrofit stack misbehaves, **who is at fault**? Cruise, the OEM, the owner?
- **Hidden vehicle state:** **ABS**, stability control, and ECU software differ by trim and change with **silent manufacturer updates**; each retrofit is a moving target.
- **Long tail:** An “infinite” list of edge cases across variants- not acceptable when lives are on the line.
- **Business fragmentation:** The team assumed **three** popular models would cover ~**80%** of San Francisco; surveys showed drivers use **20–50** models. Each variant needs custom **hardware**, **calibration**, and ongoing maintenance— many small “butterflies,” not one scalable SKU.

Retrofit was a strong **story for investors and press** (scale without Google’s spend) but a weak **safety** and **unit economics** model.

## Fleet economics and monetization

**Self-driving fleets** economics hinge on:

- **Build cost** (vehicle + **sensors** + compute).
- **Lifetime mileage** (100k vs **2M** miles changes everything).
- **Utilization**— like airlines, maximize **hours/day** generating revenue.
- **Revenue model**.

Near-term opportunities once capability exists:

- **Ride-hailing**— clear demand, privacy, consistency, safety vs human-driven **Uber/Lyft**; crowded market.
- **Delivery** (parcels, food, groceries)— active in the prior 6–12 months.

**Zero monetization** until a fleet of **driverless** vehicles matches or beats **human** driving.

## From prototype to production: the “grind”

After **~4.5 years** focused on driverless tech, Vogt felt core **maneuvers** (left turns, double-parked obstacles, **construction**) had **scaffolding** early. The gap is not one scenario at **100%** failure. It is beating **human** performance on **edge cases** (humans excel at rare events; machines historically do not).

Process: **continuous improvement** —catalog uncomfortable or risky events, rework subsystems, repeat. **Thousands** of small fixes: more **deep learning**, **test coverage**, simulated scenarios. The **unsexy** phase from **prototype** to **production**— scaling the grind with human experts and **ML**.

## Building successful startups

Vogt co-founded **Justin.tv** / **Twitch** and **Cruise** (each acquired for **~$1B**)— with **survivor bias** caveats.

Common threads:

1. **Passion** for core technology—thinking about problems at night; stamina through startup **ups and downs** as if the work would be done unpaid.
2. **Strong people**— logical, high-integrity co-founders (**Dan Kan** at Cruise; Justin.tv/Twitch co-founders).
3. **Persistence**— conviction on the original thesis plus “unsexy” work (**finance**, **HR**, **ops**). Progress daily; failures are often **self-inflicted** or quitting, not only competitors or runway.
