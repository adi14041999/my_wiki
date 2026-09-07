# Jul 2019. [Chris Urmson- Self-Driving Cars at Aurora, Google, CMU, and DARPA | Lex Fridman Podcast no. 28](https://www.youtube.com/watch?v=Tj6NOfdfa4o&list=PLypW8HeibkIJMB_Yl3S5KAr69VBF3OPIa)
22 Jul 2019.

## The DARPA Challenges

Urmson was a graduate student at Carnegie Mellon and technical director of the CMU team for the DARPA Grand Challenge and the DARPA Urban Challenge, under Red Whittaker.

The high order bit from those races was simply that **it could be done**. At the time the problem was marked as very nearly impossible, and CMU was the only robotics institute around, so falling on their faces would have been embarrassing. He credits a certain benefit to **naivete**: not knowing how hard something really is lets you try things that wiser people would not.

## Red Whittaker and leadership

Two lessons Urmson takes from Whittaker:

1. **Go and try the really hard things:** That is where the opportunity is.
2. **See people for who they can be, not who they are:** Whittaker would look at undergraduates and graduate students and empower them to lead and take real responsibility, where another person would see "just a graduate student, what could they know?" Trust, verify, and have confidence in what people can become.

## Technical evolution of the systems

**Grand Challenge: HD mapping:** HD mapping (high-definition mapping) means building a centimetre-to-decimetre accurate 3D model of a road environment ahead of time, so the vehicle isn't discovering the world from scratch as it drives. Before it, most off-road robotics work had been done with no real prior model of what the vehicle would encounter.

**Urban Challenge: multi-beam lidar:** Generating a high-resolution, mid- to long-range 3D model of the world around the vehicle was game-changing for understanding the surroundings.

In parallel, several converging techniques had their day in the sun. **Bayesian estimation** and **SLAM** had been dominant in robotics research.

SLAM stands for Simultaneous Localization and Mapping. It's the problem a robot faces when it's dropped into an unknown environment and has to answer two questions that depend on each other:

Where am I? (localization)— which requires a map to be positioned against.
What does this place look like? (mapping)— which requires knowing where you were when you took each measurement.

Neither can be solved first. That circularity is what makes SLAM interesting: you have to estimate the robot's trajectory and the structure of the world jointly, from the same stream of noisy sensor data.

Urmson is careful about the terminology: they were not really doing SLAM in real time. They had a road map ahead of time, so they were doing **localization** against a model of the world, using lidar or cameras depending on the team. The step that mattered was moving away from naively trusting GPS/INS. Nothing there was innovative as localization research. What was notable was seeing that technology be *necessary* in a real application on a big stage.

### Perception at the Urban Challenge

Behind where it is today, but the core was there. The team tracked vehicles at 100+ metre range because they had to merge with traffic, used Bayesian estimates of vehicle state, and had to predict where another vehicle would be a few seconds into the future.

They also had to handle two things that remain central:

- **Multiple hypotheses:** A vehicle at an intersection might go straight, right, or left.
- **Interaction:** Their own behaviour would change the other operator's behaviour. They handled this in relatively naive ways, but it still had to be handled.

## What makes the real world harder

The fundamental difference is that in the real world you are doing it for real.

The Urban Challenge was a limited-complexity environment: certain actors were absent, roads were maintained, barriers kept people separated from the robots. There were no cyclists, no pedestrians, no traffic lights. And it only had to work for **60 miles**. From 2006 that sounded like a lot. From 2019, when you want a vehicle to drive half a million miles, it is a different game.

The biggest change is that the other road users are **truly unpredictable**. Most drivers behave well most of the time, but the variety of behaviour and the scale over which you must operate are far beyond what the challenges required.

## Lidar and the sensor suite

Urmson considers lidar essential— and cameras essential, and radar essential. To be genuinely robust, you need the **composition** of data from different sensors.

### Cost

Cost is the word that comes up more than any other when talking to automotive companies.

Urmson pushes back on the framing of "the cheapest sensor suite." What you actually want is a suite that is **economically viable**. After that, everything is margin and driving cost out of the system. It is a nice story that a $50 sensor beats a $500 one, but if the $500 sensor makes it work and the $50 sensor does not, who cares? Without a working system there is no economic opportunity, and without that there is no sustainable business and no path to scale.

On lidar specifically, he sees no *fundamental* expense in the technology. It will be more expensive than an imager, because CMOS and fab processes are dramatically more scalable than mechanical ones, but substantial cost can still be driven out. And with the right business model you can absorb more bill-of-materials cost, because the sensor suite is delivering the value.

## Level 2 autonomy and the human factor

!!! note "SAE levels of driving automation"
    **Level 1** automates one axis (steering *or* throttle/brake). **Level 2** automates two axes— steering *and* throttle/brake while the human remains responsible for monitoring at all times. **Level 3** lets the driver disengage attention under specific conditions but requires them to take back control when asked. **Level 4** is full autonomy within a defined operational domain, with no expectation that a human takes over.

Urmson is careful to be precise here, because the point gets twisted. **Active safety systems are important technology** that we should be pursuing and integrating into vehicles; there is a near-term opportunity to reduce accidents and fatalities. Variants of Level 2 that genuinely support the driver should be encouraged.

The real challenge is the **human factors** part. The public's misconception of what the capability set actually is, and the trust they place in it accordingly. He is incrementally more concerned about Level 3 systems, and about how Level 2 systems are marketed and delivered.

His first belief is that **people will over-trust the technology**. He cites a spate of reports of people sleeping in their Teslas. He says that it is not a self-driving car and is not intended to be, and people who treat it as one will at some point be killed or hurt others.

### Why the Level 2 path diverges

The second belief is economic, and it is the more interesting argument. **The technology path for driver-assistance systems diverges from the path to true self-driving**, so you cannot simply increment your way from one to the other.

### Can a Level 2 vehicle be used without over-trust?

Urmson does not think so. If people truly understood and internalized the risks, sure, but that world does not exist.

His illustration: someone drives up and down the 101 every day for a month on a Level 2 system and it works every time. Even knowing the statistics, personal experience is compelling. But 60 miles a day for 30 days is roughly 1,800 miles, a drop in the bucket against the ~85 million miles between fatalities. Their personal experience gives them no real estimate of the risk, and they will trust it anyway.

## Metrics

Urmson think a useful metric exists, but that it will not be a single number. The approach he describes is to decompose driving into **capabilities** (detecting traffic lights, safely making a left turn across traffic) and ask what the human failure rate is for each, then demonstrate to themselves, to regulators, and to the public that the system beats it. These individual metrics can tell a compelling story.

Ultimately what is cared about is lives saved, injuries reduced, and casualty dollars. But an event per 85 million miles is statistically difficult to compare directly even at the scale of the US. So he points to the aviation approach of an **event pyramid**: crashes at the top, then injuries, then near-misses, then violations of operating procedure, with a statistical model relating the frequent low-severity events to the rare high-severity ones.

## Why urban and suburban before highway

Aurora's focus is moderate-speed urban and suburban environments rather than trucking, which is the opposite of the common intuition.

The intuition says freeways are easier. Everyone going the same direction, wider lanes. Urmson thinks that intuition is fine as far as it goes, but it is about the *average* case, and "we don't really care about most of the time. We care about all of the time." A truck at 70 mph with a 70,000-pound load carries an enormous amount of kinetic energy, so when it goes wrong it goes very wrong. Those failures also occur **more rarely**, so you learn more slowly. Freeway driving is therefore incrementally more difficult, not easier.

In a moderate-speed urban environment, two vehicles colliding at 25 mph is not good but probably everyone walks away, and the situations where that could occur happen frequently. So you **learn more rapidly, at lower risk for everyone**, while delivering real value to people getting from place to place. Once that is solved, the freeway case largely falls out.
