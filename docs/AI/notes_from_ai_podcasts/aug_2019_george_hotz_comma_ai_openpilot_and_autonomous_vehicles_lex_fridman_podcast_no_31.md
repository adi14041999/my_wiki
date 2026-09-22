# Aug 2019. [George Hotz- Comma.ai, OpenPilot, and Autonomous Vehicles | Lex Fridman Podcast no. 31](https://www.youtube.com/watch?v=iwcYp-XT7UI&list=PLypW8HeibkIJMB_Yl3S5KAr69VBF3OPIa)
5 Aug 2019.

## Simulation

Hotz thinks we probably are living in a simulation, but that it may be **unfalsifiable**. If the builders formally proved that no information can get in or out, and designed their hardware so nothing inside can push it out of spec, you could never tell— the same way a well-designed virtual machine cannot be detected from inside.

## Hacking

The August 2007 iPhone unlock that made him famous was **a hardware hack**. He had read *Gray Hat Hacking* in 2006 and understood that these powers let you control the world, but he knew electronics, not software. The unlock involved opening the phone and pulling an address line high. The code he released alongside it he calls atrocious— a broken state machine in C. He did not know how to program.

He learned by **building the same tool four times**. He wanted to emulate and visualize ARM binaries so he could single-step through the iPhone's boot ROM and bootloader, where there was no debugger. The first version was terrible, the second terrible, the third tolerable, and the fourth— **QIRA** (written during a Google internship in 2014) was finally usable. It is a *timeless debugger*: rather than setting a watch on a variable, you click it and see every moment it was changed or accessed, and rewind as easily as you step forward. It is not widely used because it does not scale to a 200 MB binary like Chrome, but it is excellent for small code like boot ROMs.

**Capture the flag** competitions were his training ground. A vulnerable program listens on a socket, a file called `flag` sits on the server, you write an exploit that gets you a shell, read the flag, and submit it for points. On how much harder real systems have become: jailbreaking an iPhone used to take one exploit; now it takes about **nine chained together**.

## How he programs

At 22 he went to Carnegie Mellon and took the hardest CS courses to see whether he had missed anything. Operating systems and compilers were two of the best classes of his life. He wrote the OS in C and the compiler in Haskell. And he picked up Python that semester for CTFs, where you use whatever language is fastest to write in.

He spent five months on Google's **Project Zero**, the offensive security team, when it started in 2015. The idea: vendors often do not care about vulnerabilities reported through bug bounties, so Project Zero reports one, gives the vendor **90 days**, and publishes the zero-day whether it is fixed or not. He thinks it has done a lot to move the industry forward.

The code he is proudest of per line is a **200-line distributed file system** he wrote for comma, using nginx for the volume servers and a small master server in Go.

On languages: openpilot is migrating toward C, and he now understands why **Python does not work for large codebases**. On JavaScript, the ecosystem is unbelievably confusing, but it is fast and in the browser and the worst languages never die. All comma's debugging tools are JavaScript, because **the web is the best UI toolkit by far**.

## How comma started

A friend told him the coolest applied AI problem was self-driving cars and introduced him to Elon Musk, who was looking to replace Mobileye in Autopilot 1 (AP1). The proposed contract: deliver a vision system at Mobileye's level of performance and get **$12 million, minus $1 million for every month it took**.

The goal was never level 5. Mobileye's output was two lanes and the position of the lead car, and he figured he could gather a dataset and train a net in weeks. He had something he judged the same quality **in about three months**, by driving a borrowed Tesla on AP1 and then his own system and comparing.

## Lane keeping is the product

This is Hotz's central commercial claim, and he states it bluntly: **lane centering is the only feature in autonomous vehicles today that actually adds value to people's lives**.

openpilot, run for an hour on a stretch of highway, will never touch a lane line. And the value is **removing the stress of staying in lane**.

### The hardware

The box is a phone in a plastic case: a Snapdragon 820, a forward-facing camera, a driver-monitoring camera, and a CAN transceiver called the **panda** that talks to the phone over USB and to the car on three CAN buses— the radar bus, the main car bus, and a proxied camera bus. It works because the supported cars (45 models by then, mostly Hondas and Toyotas, plus GM and Subaru) already have lane keeping assist. Lane keeping assist can mean anything from a nudge after you have crossed the line by a foot to Tesla-style centring. The 2020 Corolla is the best car for it, because its actuator has less lag.

They had moved from camera-only to **fusing in the car's radar**.

## Driver monitoring and the takeover

openpilot will not be 1.0 until it has **driver monitoring you cannot cheat**. It currently tracks head pose (the camera cannot resolve eyes well) and the next steps are detecting a phone in frame and detecting sleep. Part of the value is simply psychological: a monitor that is always there reminds you to pay attention.

He is emphatic that Musk's claim that driver monitoring is unnecessary is **stupid**. When these systems get to the point where they only mess up once every thousand miles, you absolutely need it.

openpilot disengages on gas or brake, but not on steering. Autopilot, by contrast, requires a double press to engage, and to cancel you must either find a cancel button nobody knows about, press the brake (often not what you want), or wiggle the wheel. Its hands-on-wheel check requires you to apply just enough torque to register but not enough to disengage it. Switching in and out should be nearly free.

He admires GM's **Super Cruise** once it is engaged. comma bought a Cadillac so everyone in the office could experience what they were aiming for. But the transitions are poor. comma renamed its driver-monitoring packet to *driver state* to sit alongside *car state*: a system should be transparent about what it sees of the driver, as Tesla is about what it sees of the road.

On whether you could detect drunkenness with computer vision, he thinks yes, probably combined with how someone is controlling the car. Fix drunk, distracted and asleep and you have fixed a great deal.

## Why perception and planning cannot be separated

This is the technical core of the episode.

Waymo-style stacks divide the problem: a perception system produces a description of the scene, and a planner reasons over it. Hotz's claim is that **there is no human-understandable state vector that separates perception from planning**.

Between localization and planning there is one (three degrees of position, three of orientation, and their derivatives). But try writing the output of a perception system: a list of cars, a list of pedestrians, the drivable area. It can never be complete.

> If your perception system output can be written in a spec document, it is incomplete.

comma's perception output is a **1,024-dimensional vector of who knows what**. Lex restates it well: the conventional stack converts the scene into a chessboard and reasons over the chessboard, and a lot of the world does not fit on it.

The conclusion he draws is sweeping. If you accept this, then what Waymo and Cruise are doing is currently impossible.

### Learning lane changes

comma's method is simple. The model has an input bit saying whether it is doing a lane change. An automatic labeller marks lane changes in the training data by detecting lane-line crossings (about 95% accurate, which is good enough).

!!! note "How training with the bit actually works"
    **The basic training setup, without the bit:** Start with plain behavioral cloning, which is supervised learning on human driving logs:

    * Input: camera frames, plus some vehicle state such as speed.
    * Target: what the human actually did next, meaning the path the car followed over the next few seconds.
    * Loss: how far the predicted path is from the real one.

    **Adding the bit.** The fix is to give the network the information it's missing: a bit that says whether the driver intends to change lanes.

    ```
    inputs  = (camera_frames, vehicle_state, lane_change_bit)
    target  = future_path_actually_driven
    loss    = distance(model(inputs), target)
    ```

    During training, set the bit from the data:

    * bit = 1 on frames that belong to a lane change.
    * bit = 0 everywhere else.

    Now the two cases that looked identical are different inputs:

    * (this scene, bit = 0) → the human stayed in lane
    * (this scene, bit = 1) → the human moved over

    The ambiguity is gone, so the network can learn a sharp answer for each. The bit doesn't say how to change lanes. It only says that one is happening, and the network learns how from thousands of human examples. That's why Hotz can claim the result adapts to the situation: in heavy traffic it has seen humans change lanes one way, and on an empty road another.

    At test time you set the bit yourself. The driver flicks the turn signal, openpilot sets the bit to 1, and the model outputs the kind of path a human would drive when changing lanes in that scene.

On learning from bad drivers, he offers a lovely hypothesis with some data behind it:

> All good drivers are good in the same way, and all bad drivers are bad in different ways.

Good drivers form a cluster; bad drivers scatter. So a net trained on everyone learns the cluster, and the random behaviour washes out as noise. There may be four good lane-change modes, or twenty, and which one to use depends on the scene rather than the driver. The hope is that the distribution is a nice Gaussian rather than bimodal.

He thinks Karpathy's "software 2.0" strategy at Tesla is exactly right, and that Navigate on Autopilot was written by someone else and hacked on top. Karpathy, he expects, will look at it and ask why anyone hand-coded a lane-change policy full of magic numbers.

## The three driving problems

Hotz divides driving into three problems.

1. **Static**— you are the only car on the road. This is solvable entirely with mapping and localization, which is why automated farms work: you can statically schedule tractors, like scheduling processes, so their paths never cross. Maps only help with this problem. It is not easy for the whole world. Tesla drifting out of lane is a failure at exactly this. But it is tractable with a good localizer. With lidar you get to a centimeter, without it about ten; what matters is never being way off without knowing it.
2. **Dynamic**— a car stopped at a red light. It cannot be in your map, because you do not know whether it will be there. You must detect it, predict where it will move, and respond.
3. **Counterfactual**— other agents react to *you*.

For the third, the only route he sees is **reinforcement learning on the real world**, since the other agents are humans.

The learning signal comes free. openpilot, with about 700 daily and 1,000 weekly active users, makes **tens of thousands of mistakes a week**.

## Lidar is a crutch

On Musk's claim, Hotz says it is one of the times Musk says something completely, obviously true: **of course lidar is a crutch, and not even a good one**. The lidar companies are mostly using it for **localization**, not perception.

- **Waymo** is by far the furthest along technically, with perhaps a three-year lead. But he argues it has spent too much money to recoup those three years, because **self-driving fleets have no network effect**.
- Uber has a two-sided market: it needs drivers and riders simultaneously, and switching costs protect it. A rival taking a 5% cut instead of 10% would not win, because drivers and riders would have to switch together. A robotaxi fleet needs only capital and riders. Anyone with a chequebook can buy off-the-shelf cars and blanket a city, like **scooters** (which is why there are ten scooter companies in a race to the bottom).
- **Cruise**'s nice thing is that buying it was a great move for GM: for a billion dollars, GM bought an **insurance policy** against a Waymo monopoly, capping it at roughly three years. He would be very surprised if Cruise leapfrogged Waymo.

He adds that Cruise, Waymo, Aurora and Zoox are effectively the same stack. They all descend from the DARPA Urban Challenge codebase.

!!! note "The opposite view"
    [Chris Urmson](jul_2019_chris_urmson_self_driving_cars_at_aurora_google_cmu_and_darpa_lex_fridman_podcast_no_28.md#lidar-and-the-sensor-suite) (whom Hotz names as someone he deeply respects) argues nearly the reverse on each point: lidar is essential, [driver assistance and full autonomy are diverging technologies](jul_2019_chris_urmson_self_driving_cars_at_aurora_google_cmu_and_darpa_lex_fridman_podcast_no_28.md#why-the-level-2-path-diverges) so you cannot increment from one to the other, and [urban driving should come before highways](jul_2019_chris_urmson_self_driving_cars_at_aurora_google_cmu_and_darpa_lex_fridman_podcast_no_28.md#why-urban-and-suburban-before-highway). The two episodes read well against each other.

## Safety model

comma's line is: **we are proud to be level 2**. Level 4 will not arrive as a magical over-the-air update to current hardware.

The model has three rules:

1. The driver must be paying attention at all times.
2. The driver must be able to take control easily at all times: gas, brake or cancel returns full manual control.
3. The car will never react so quickly that you cannot respond in time (about one second) enforced by **torque, braking and acceleration limits**. comma's torque limit is far lower than Tesla's; Autopilot can jerk the wheel hard.

The code is open source and, he believes, now **MISRA C** compliant. He has come to respect the automotive standards, which were clearly written by very smart computer scientists.

**Security**, he argues, is a special case of safety. Safety is caution tape around a hole; security is a ten-foot fence with barbed wire. If your system is unreliable it is certainly not secure. A car should always do something safe using its **local sensors**, which should be hardwired. So:

- **V2V** (vehicle-to-vehicle) is a terrible idea, because it depends on both parties communicating correctly.
- **V2I** (vehicle-to-infrastructure) is fine for non-safety-critical things. Waze routing around traffic is already V2I.
- **Teleoperation** does not work if the safety design requires a constant link.

Adversarial examples exist for humans too. Put a black bag over a stop sign and people run it.

## The business

Burn was about **$200k a month** against revenue of about **$100k**, all from selling hardware online, so they needed to double revenue (though they had not tried hard). The near-term milestone was profitability by mid-2020, followed by revisiting a consumer product that would be done properly this time.

The long-term plan, known since 2017: **become a car insurance company**. If comma makes driving twice as safe and has the best data on who is statistically the safest driver, it can refuse to insure people it sees driving unsafely. That bifurcates the market— the only people who cannot get comma insurance are bad drivers, whom Geico can take on at very high premiums, while comma's are very low.

He is also taking on the problem directly (unlike [Kyle Vogt, who abandoned the aftermarket retrofit](feb_2019_kyle_vogt_cruise_automation_lex_fridman_podcast_no_14.md#why-retrofit-failed) partly because supporting 20–50 car models seemed impossible). comma supports 45, with alerts handling each manufacturer's emergency braking, and has ten million miles of data.

The business models he most respects are the ones **creating value today**: Nauto, which sells fleet owners a driver-monitoring camera for about $40 a month that reduces accidents by roughly 18%.

## What "winning" means

He has said the meaning of life is to win, and was criticized for it. What he means is not a yacht. He is an agent put into a world without knowing its purpose, and he takes a goal from **Jürgen Schmidhuber**: explore so as to maximize the rate of improvement. Maybe he will be given a real purpose, or decide one— and then he will know what the game is and how to win it. For now he is trying to discover the reward function while maximizing it under uncertainty.
