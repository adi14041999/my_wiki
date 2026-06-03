# Feb 2019. [Kyle Vogt- Cruise Automation | Lex Fridman Podcast no. 14](https://www.youtube.com/watch?v=YUYagvESisE&list=PLypW8HeibkIJMB_Yl3S5KAr69VBF3OPIa)
7 Feb 2019.

it was all heuristic space and so like old-school image processing and I think extracting you know yellow lane
9:159 minutes, 15 secondsmarkers out of an image of a road is one of the problems that lends itself reasonably well to those heuristic based
9:239 minutes, 23 secondsmethods you know like just do a threshold on the color yellow and then try to fit some lines to that using a Hough transform or something and then go
9:319 minutes, 31 secondsfrom there traffic like detection and then stop signs detection red yellow green and I think you can you could I mean if you wanted to do a full I was just trying to
9:409 minutes, 40 secondsmake thing that would stay in between the lanes on a highway but if you wanted to do the full the full you know set of
9:489 minutes, 48 secondscapabilities needed for a driverless car I think you could and we done this at cruise you know in the very first days you can start off with a really simple
9:559 minutes, 55 secondsyou know human written heuristic just to get the scaffolding in place for your system traffic light detection probably a really simple you know color threshold
10:0310 minutes, 3 secondsinjustice system up and running before you migrate to you know a deep learning based technique or something else and you know back in when I was doing this
10:1210 minutes, 12 secondsmy first one it was on Pentium 203 233 megahertz computer in it and I I think I wrote the first version in basic which
10:2010 minutes, 20 secondsis like an interpreted language it's extremely slow because that's the thing I knew at the time and so there was no no chance at all of using there was no
10:2810 minutes, 28 secondscomputational power to do any sort of reasonable deep nets like you have today so I don't know what kids these days are doing our kids these days you know at
10:3710 minutes, 37 secondsage 13 using neural networks in their garage I mean I also I get emails all the time from you know like 11 12 year
Chapter 8: Deep Learning
10:4410 minutes, 44 secondsold saying I'm having you know I'm trying to follow this tensorflow tutorial and I'm having this problem
10:4910 minutes, 49 secondsand their general approach in the deep learning community is of extreme
10:5710 minutes, 57 secondsoptimism of as opposed to you mentioned like heuristics you can you can separate the autonomous driving problem into
11:0411 minutes, 4 secondsmodules and try to solve it sort of rigorously or you just do it end to end and most people just kind of love the idea that you know us humans do a tenth
11:1311 minutes, 13 secondsand we just perceive and act we should be able to use that do the same kind of thing when you're on that's and that that kind of thinking you don't want to
11:2111 minutes, 21 secondscriticize that kind of thinking because eventually they will be right yeah and so it's exciting and especially when they're younger to explore that is a
11:2911 minutes, 29 secondsreally exciting approach but yeah it's it's changed the the language the kind of stuff you turned green with it it's
11:3711 minutes, 37 secondskind of exciting to see when they seniors grow up yeah I can only imagine if you if your starting point is you know Python and tensorflow at age 13
11:4611 minutes, 46 secondswhere you end up you know after 10 or 15 years of that that's that's pretty cool because of github because this they're tools for solving most of the
Chapter 9: Entrepreneurship
11:5411 minutes, 54 secondsmajor problems and artificial intelligence are within a few lines of code for most kids and that's incredible to think about also on the
12:0212 minutes, 2 secondsentrepreneurial side and and and at that point was there any thought about entrepreneurship before you came to
12:1112 minutes, 11 secondscollege is sort of doing your building this into a thing that impacts the world on the large scale yeah I've always
12:1812 minutes, 18 secondswanted to start a company I think that's you know just a cool concept of creating something and exchanging it for value or
12:2612 minutes, 26 secondscreating value I guess so in high school I was I was so trying to build like you know a servo motor drivers little circuit boards and sell them online or
12:3412 minutes, 34 secondsother other things like that and certainly knew at some point I wanted to do a startup but it wasn't really I'd say until college until I felt like I
12:4312 minutes, 43 secondshad the I guess the right combination of the environment the smart people around you and some free time and a lot of free
12:5112 minutes, 51 secondstime at MIT

Chapter 11: AI in Autonomous Vehicles
14:0514 minutes, 5 secondsvehicles in terms of artificial intelligence evolve in this moment I mean you know like you said from the 80s has been autonomous vehicles but really
14:1414 minutes, 14 secondsthat was the birth of the modern wave the the thing that captivated everyone's imagination that we can actually do this
14:2014 minutes, 20 secondsso what how were you captivated in that way so how did your view of autonomous vehicles change at that point I'd say at
14:2914 minutes, 29 secondsthat point in time it was it was a the curiosity as in like is this really possible and I think that was generally
14:3714 minutes, 37 secondsthe spirit and the the purpose of that original DARPA Grand Challenge which was to just get a whole bunch of really
14:4514 minutes, 45 secondsbrilliant people exploring the space and pushing the limits and and I think like to this day that DARPA challenge with
14:5314 minutes, 53 secondsits you know million dollar prize pool was probably one of the most effective you know uses of taxpayer money dollar
15:0115 minutes, 1 secondfor dollar that I've seen you know because that that small sort of initiative that DARPA put put out sort
15:0915 minutes, 9 secondsof in my view was the catalyst or the tipping point for this this whole next wave of autonomous vehicle development so that was pretty cool so let me jump

the three things where it had to be something where the technology itself determines the success of the product like hard really juicy technology
24:3124 minutes, 31 secondsproblems because that's what motivates me and then it had to have a direct and positive impact on society in some way so an example would be like you know
24:3924 minutes, 39 secondshealthcare self-driving cars because they save lives other things where there's a clear connection to somehow improving other people's lives and the last one is it had to be a big business
24:4624 minutes, 46 secondsbecause for the positive impact to matter it's got to be a large scale scale yeah and I was thinking about that for a while and I made like I tried
24:5424 minutes, 54 secondswriting a gmail clone and looked at some other ideas and then it just sort of light bulb went off like self-driving cars like that was the most fun I had
25:0125 minutes, 1 secondever had in college working on that and like well what's the state of the technology has been ten years maybe maybe times have changed and maybe now
25:0825 minutes, 8 secondsis the time to make this work and I poked around and looked at the only other thing out there really at the time was the Google self-driving car project
25:1525 minutes, 15 secondsand I thought surely there's a way to you know have an entrepreneur mindset and sort of solve the Minimum Viable Product here and so I just took the
25:2425 minutes, 24 secondsplunge right then in there and said this this is something I know I can commit ten years to it's the probably the greatest applied AI problem of our generation it's right and if it works
25:3225 minutes, 32 secondsit's going to be both a huge business and therefore like probably the most positive impact I can possibly have on the world so after that light bulb went
25:4025 minutes, 40 secondsoff I went all in on crews immediately and got to work did you have an idea how to solve this problem which aspect of
Chapter 18: How to solve the problem
25:4725 minutes, 47 secondsthe problem to solve you know slow like what we just had Oliver for voyage here slow-moving retirement communities urban
25:5625 minutes, 56 secondsdriving highway driving did you have like did you have a vision of the city of the future or you know the
26:0326 minutes, 3 secondstransportation is largely automated that kind of thing or was it sort of more fuzzy and gray area than that my
26:1226 minutes, 12 secondsanalysis of the situation is that Google is putting a lot it had been putting a lot of money into that project that a
26:1926 minutes, 19 secondslot more resources and so and they still hadn't cracked the fully driverless car you know this is 20 2013
26:2826 minutes, 28 secondsI guess so I thought what what can I do to sort of go from zero to you know significant scale so I can actually
26:3626 minutes, 36 secondssolve the real problem which is the driverless cars and I thought here's the strategy we'll start by doing a really simple problem or solving a really
26:4426 minutes, 44 secondssimple problem that creates value for people so eventually ended up deciding on automating highway driving which is
26:5226 minutes, 52 secondsrelatively more straightforward as long as there's a backup driver there and I'll you know the go-to-market will be able retrofit people's cars and just
27:0027 minutessell these products directly and the idea was we'll take all the revenue and profits from that and use it to do the
27:0727 minutes, 7 secondssocial reinvest that in research for doing fully fabulous cars and that was the plan the only thing that really changed along
27:1527 minutes, 15 secondsthe way between then and now is we never really launched the first product we had enough interest from investors in enough of a signal that this was something that
27:2427 minutes, 24 secondswe should be working on that after about a year of working on the highway autopilot we had it working you know on a prototype stage but we just completely
27:3127 minutes, 31 secondsabandoned that and said we're gonna go all in on driverless cars now is the time can't think of anything that's more exciting and if it works more impactful
27:3927 minutes, 39 secondsso we're just gonna go for it the idea of retrofit is kind of interesting yeah being able to it's how you achieve scale it's a really interesting idea is it's
Chapter 19: Retrofit
27:4827 minutes, 48 secondssomething that's still in the in the back of your mind as a possibility not at all I've come full circle on that one
27:5627 minutes, 56 secondstrying to build a retrofit product and I'll touch on some of the complexities of that and then also having been inside in OEM and seeing how things work and
28:0528 minutes, 5 secondshow a vehicle is developed and validated when it comes to something that has safety critical implications like controlling the steering and the other
28:1328 minutes, 13 secondscontrol inputs on your car it's pretty hard to get there with with a retrofit or if you did even if you did it it creates a whole bunch of new
28:2128 minutes, 21 secondscomplications around liability or how did you truly validate that or you know something in the base vehicle fails and causes your system to fail whose fault
28:2928 minutes, 29 secondsis it or if the cars anti-lock brake systems or other things kick in or the software has been it's different in one version
28:3728 minutes, 37 secondsof the car you retrofit versus another and you don't know because the manufacturer has updated it behind the scenes there's basically an infinite list of longtail issues that can get you
28:4628 minutes, 46 secondsand if you're dealing with a safety critical product that's not really acceptable that's a really convincing summary of why it's really challenging but I didn't at the time so we tried it
28:5428 minutes, 54 secondsanyway but it's a pitch also at the time it's a really strong one yes that's how you achieve scale and that's how you beat the current the the leader at the
29:0229 minutes, 2 secondstime of Google or the only one in the market the other big problem we ran into which is perhaps the biggest problem from a business model perspective is we
29:1129 minutes, 11 secondshad kind of assumed that we'd we started with an Audi s4 as the vehicle we retrofitted with his highway driving capability and we had kind of assumed
29:2029 minutes, 20 secondsthat if we just knock out like three make and models of vehicle that'll cover like eighty percent of a San Francisco market doesn't everyone there drive I
29:2729 minutes, 27 secondsdon't know a BMW or a Honda Civic or one of these three cars and then we surveyed our users we found out that it's all over the place we would to get even a
29:3429 minutes, 34 secondsdecent number of units sold we'd have to support like you know 20 or 50 different models and each one is a little butterfly that takes time and effort to
29:4229 minutes, 42 secondsmaintain you know that retrofit integration and custom hardware and all this so is it there's a tough business so GM manufactures and sells over nine
Chapter 20: Detroit vs Silicon Valley
29:5229 minutes, 52 secondsmillion cars a year and what you with crews are trying to do some of the most cutting-edge innovation in terms of
30:0130 minutes, 1 secondapplying AI and so hot out of those you've talked about a little bit before but it's also just fascinating to me we'll work a lot of automakers you know
30:1030 minutes, 10 secondsthe difference between the gap between Detroit and Silicon Valley let's say just to be sort of poetic about it I guess what how do you close
30:1730 minutes, 17 secondsthat gap how do you take GM into the future where a large part of the fleet would be autonomous perhaps I want to
30:2530 minutes, 25 secondsstart by acknowledging that that GM is made up of you know tens of thousands of really brilliant motivated people who want to be a part of the future and so
30:3330 minutes, 33 secondsit's pretty fun to work within the attitude inside a car company like that is you know embracing this this transformation and change rather than
30:4130 minutes, 41 secondsfearing it and I think that's a testament to the leadership at GM and that's flown all the way through to to everyone you talk to even the people in this in blue
30:4830 minutes, 48 secondsplants working on these cars so that's really great so that starting from that position makes a lot easier so then when
30:5630 minutes, 56 secondsthe the people in San Francisco at Cruz interact with the people at GM at least we have this common set of values which is that we really want this stuff to
31:0431 minutes, 4 secondswork because we think it's important and we think it's the future not to say you know those two cultures don't clash they absolutely do there's
31:1231 minutes, 12 secondsdifferent different sort of value systems like in a car company the thing that gets you promoted and so the reward
31:1931 minutes, 19 secondssystem is following the processes delivering the the program on-time and on-budget so any sort of risk-taking is
31:2831 minutes, 28 secondsdiscouraged in many ways because if a program is late or if you shut down the plant for a day it's you know you can
31:3631 minutes, 36 secondscount the millions of dollars that burn by pretty quickly whereas I think you know most Silicon Valley companies and
31:4431 minutes, 44 secondscrews in the methodology we were employing especially around the time of the acquisition the reward structure is
31:5131 minutes, 51 secondsabout trying to solve these complex problems in any way shape or form or coming up with crazy ideas that you know
31:5831 minutes, 58 seconds90% of them won't work and and so so meshing that culture of sort of continuous improvement and experimentation with one where
32:0632 minutes, 6 secondseverything needs to be you know rigorously defined upfront so that you never slip a deadline or miss a budget was a pretty big challenge and that
32:1432 minutes, 14 secondswe're over three years in now after the acquisition and I'd say like you know the investment we made in figuring out
32:2132 minutes, 21 secondshow to work together successfully and who should do what and how we bridge the gaps between these very different systems and way of doing engineering
32:2832 minutes, 28 secondswork is now one of our greatest assets because I think we have this really powerful thing but for a while it was both both GM and crews were very steep
32:3632 minutes, 36 secondson the learning curve yes I'm sure it was very stressful it's really important work because that's that's how to revolutionize the transportation it
Chapter 21: The culture gap
32:4332 minutes, 43 secondsreally to revolutionize any system you know you look at the healthcare system or you look at the legal system I have people like lawyers come up to me all
32:5132 minutes, 51 secondsthe time like everything they're working on can easily be automated but then that's not a good feeling yeah that was it's not a good feeling but also there's
33:0033 minutesno way to automate because the the the entire infrastructure is really you know based is older and it moves very slowly
33:0833 minutes, 8 secondsand so how do you close the gap between I haven't how can I replace of course lawyers don't wanna be replaced with an
33:1533 minutes, 15 secondsapp but you could replace a lot of aspect when most of the data is still on paper and so the same thing was with automotive I mean it's fundamentally
33:2433 minutes, 24 secondssoftware so it's is basically hiring software engineers it's thinking a software world I mean I'm pretty sure nobody in Silicon Valley's ever hit a
33:3333 minutes, 33 secondsdeadline so and then it's probably true yeah and GSI is probably the opposite yeah so that's that culture gap is
33:4133 minutes, 41 secondsreally fascinating so you're optimistic about the future of that yeah I mean from what I've seen it's impressive and I think like especially in Silicon
33:4933 minutes, 49 secondsValley it's easy to write off building cars because you know people have been doing that for over a hundred years now in this country and so it seems like that's a solved problem but that doesn't
33:5733 minutes, 57 secondsmean it's an easy problem and I think it would be easy to sort of overlook that and think that you know we're Silicon
34:0534 minutes, 5 secondsValley engineers we can solve any problem you know building a car it's been done therefore it's you know it's it's it's not it's not a real
34:1234 minutes, 12 secondsengineering challenge but after having seen just the sheer scale and magnitude and industrialization that occurs inside
34:2134 minutes, 21 secondsof an automotive assembly plant that is a lot of work that I am very glad that we don't have to reinvent to make self-driving cars work and so to have
34:2934 minutes, 29 secondsyou know partners who have done that for a hundred years now these great processes and this huge infrastructure and supply base that we can tap into is
34:3634 minutes, 36 secondsjust remarkable because the scope in surface area of the problem of deploying
34:4434 minutes, 44 secondsfleets of self-driving cars is so large that we're constantly looking for ways to do less so we can focus on the things that really matter more and if we had to
34:5334 minutes, 53 secondsfigure out how to build an assemble in you know test and build the cars themselves I mean we work closely with
35:0135 minutes, 1 secondJim on that but if we had to develop all that capability in-house as well you know that that would just make make the problem really intractable I think mmm
Chapter 22: The biggest opportunity to make money
35:0935 minutes, 9 secondsso yeah just like your first entry mit DARPA challenge when there was what the motor that failed and somebody that
35:1835 minutes, 18 secondsknows what they're doing with the motor did it that would have been nice if you focus on the software and not the hardware platform yeah right so from
35:2535 minutes, 25 secondsyour perspective now you know there's so many ways that autonomous vehicles can impact Society in the next year five
35:3235 minutes, 32 secondsyears ten years what do you think is the biggest opportunity to make money in autonomous driving sort of make it a
35:4135 minutes, 41 secondsfinancially viable thing in the near-term what do you think would be the biggest impact there well the things
35:5035 minutes, 50 secondsthat that drive the economics for fleets of self-driving cars or they're sort of a handful of variables one is you know
35:5835 minutes, 58 secondsthe cost to build the vehicle itself so the material cost how many you know what's the cost of all your sensors plus the cost of the vehicle and every all
36:0536 minutes, 5 secondsthe other components on it another one is the lifetime of the vehicle it's very different if your vehicle drives one hundred thousand miles and then it falls
36:1236 minutes, 12 secondsapart versus you know two million and then you know if you have a fleet it's kind of like an airplane where or
36:2136 minutes, 21 secondsairline where once you produce the vehicle you want it to be in operation as many hours a day as possible
36:2836 minutes, 28 secondsproducing revenue and then a you know the other piece of that is how are you generating revenue I think that's kind what you're asking and I think the
36:3736 minutes, 37 secondsobvious things today are you know the ride-sharing business because that's pretty clear that there's demand for that there's existing markets you can
36:4336 minutes, 43 secondstap into and larger urban areas that kind of thing yeah yeah and and and I think that there are some real benefits
36:5136 minutes, 51 secondsto having cars without drivers compared to through the status quo for people who use ride share services today you know
36:5836 minutes, 58 secondsyou get privacy consistency hopefully significant improve safety all these benefits versus the current product but it's it's a crowded market
37:0637 minutes, 6 secondsand then other opportunities which you've seen a lot of activity in the last really in last six to twelve months is you know delivery whether that's
37:1237 minutes, 12 secondsparcels and packages food or or groceries those are all sort of I think opportunities that are that are pretty
37:2137 minutes, 21 secondsripe for these you know once you have this core technology which is the fleet of autonomous vehicles there's all sorts
37:2837 minutes, 28 secondsof different business opportunities you can build on top of that but I think the important thing of course is that there's zero monetization opportunity
37:3637 minutes, 36 secondsuntil you actually have that fleet of very capable driverless cars that are that are as good or better than humans and that's sort of where the entire industry is sort of in this holding
37:4537 minutes, 45 secondspattern right now yeah the trend achieved that baseline so but you said sort of rely not reliability consistency it's kind of interesting I think I heard
Chapter 23: Personality of the car
37:5237 minutes, 52 secondsyou say somewhere I'm not sure if that's what you meant but you know I can imagine a situation where you would get an autonomous vehicle and you know when
38:0238 minutes, 2 secondsyou get into an uber or lyft you don't get to choose the driver in a sense that you don't get to choose the personality of the driving do you think
38:0938 minutes, 9 secondsthere's a there's room to define the personality of the car the way drives you in terms of aggressiveness for
38:1638 minutes, 16 secondsexample in terms of sort of pushing the bomb the one of the biggest challenges in Toms driving is the is a trade-off
38:2338 minutes, 23 secondsbetween sort of safety and and do you think there's any room for
38:3038 minutes, 30 secondsthe human to take a role in that decision to accept the liability I guess
38:3738 minutes, 37 secondswe III wouldn't it no I'd say within reasonable bounds as in we're not gonna I think it'd be highly unlikely we did expose any nob that would let you you
38:4638 minutes, 46 secondsknow significantly increase safety risk I think that's that's just not something we'd be willing to do but I think
38:5438 minutes, 54 secondsdriving style or like you know are you gonna relax the comfort constraints slightly or things like that all of those things make sense and are
39:0139 minutes, 1 secondplausible I see all those is you know nice optimizations once again we get the core problem solved and these fleets out there but the other thing we've sort of
39:0939 minutes, 9 secondsobserved is that you have this intuition that if you sort of slam your foot on the gas right after the light turns green and aggressively accelerate you're
39:1839 minutes, 18 secondsgonna get there faster but the actual impact of doing that is pretty small you feel like you're getting there faster but so that so the same would be true
39:2539 minutes, 25 secondsfor ABS even if they don't slam there you know the pedal to the floor when the light turns green they're gonna get you they're within you know if it's a 15-minute trip within 30 seconds of what
39:3439 minutes, 34 secondsyou would have done otherwise if you were going really aggressively so I think there's this sort of self-deception that that my aggressive
39:4239 minutes, 42 secondsdriving style is getting me there faster well so that's you know some of the things I study some things I'm fascinated by the psychology of that I don't think it matters that it doesn't
Chapter 24: Emotional release
39:5139 minutes, 51 secondsget you there faster it's it's the emotional release driving is is a place being inside or a car somebody said it's
39:5939 minutes, 59 secondslike the real world version of being a troll so you have this protection this mental protection you're able to sort of yell at the world like release your
40:0740 minutes, 7 secondsanger whatever is but so there's an element of that that I think autonomous vehicles would also have to you know have giving an outlet to people but it
40:1540 minutes, 15 secondsdoesn't have to be through through through driving or honking or so on there might be other outlets but I think to just sort of even just put that aside
40:2340 minutes, 23 secondsthe baseline is really you know that's the focus that's the thing you need to solve and then the fun human things can be solved after but so from the baseline
40:3240 minutes, 32 secondsof just solving autonomous driving and you're working in San Francisco one of the more difficult cities to operate in what
40:3940 minutes, 39 secondswhat is what is the any of you currently the hardest aspect of autonomous driving
40:4540 minutes, 45 secondsand negotiated with pedestrians is that edge cases of perception is it planning is there a mechanical engineering is it
40:5440 minutes, 54 secondsdata fleet stuff like what are your thoughts on the challenge the more challenging aspects there that's a good that's a good question I think before before we
41:0341 minutes, 3 secondsgo to that though I just wanted I like what you said about the psychology aspect of this because I think one observation I made is I think I read somewhere that I think it's maybe
41:1141 minutes, 11 secondsAmericans on average spend you know over an hour a day on social media like staring at Facebook and so that's just
41:1841 minutes, 18 secondsyou know 60 minutes of your life you're not getting back and it's probably not super productive and so that's 3,600
41:2441 minutes, 24 secondsseconds right and that's that's time you know it's a lot of time you're giving up and if you compare that to people being
41:3241 minutes, 32 secondson the road if another vehicle whether it's a human driver or autonomous vehicle delays them by even three seconds they're laying in on the horn
41:4141 minutes, 41 secondsyou know even though that's that's you know one one thousandth of the time they waste looking at Facebook every day so there's there's definitely some you know
41:4841 minutes, 48 secondspsychology aspects of this I think that are pre interesting road rage in general and then the question of course is if everyone is in self-driving cars do they even notice these three-second delays
41:5641 minutes, 56 secondsanymore because they're doing other things or reading or working or just talking to each other so it'll be interesting to see where that goes in a certain aspect people people need
42:0542 minutes, 5 secondsto be distracted by something entertaining something useful inside the car so they don't pay attention to the external world and then and then and it
42:1242 minutes, 12 secondscan take whatever psychology and bring it back to Twitter and then focus on that as opposed to sort of interacting
42:2042 minutes, 20 secondssort of putting the emotion out there into the world so it's a it's an interesting problem but baseline autonomy I guess you could say
42:2742 minutes, 27 secondsself-driving cars you know at scale will lower the collective blood pressure of society probably by a couple points yeah without all that road rage and stress so
42:3542 minutes, 35 secondsthat's a good good externality so back to your question about the technology in the the I guess the biggest problems and
42:4342 minutes, 43 secondsI have a hard time answering that question because you know we've been at this like specifically focusing on driverless cars and all the technology needed to
42:5242 minutes, 52 secondsenable that for a little over four and a half years now and even a year or two in I felt like we had
43:0043 minutescompleted the functionality needed to get someone from point A to point B as in if we need to do a left turn maneuver or if we need to drive around a you know
43:0843 minutes, 8 secondsa double parked vehicle into oncoming traffic or navigate through construction zones the the scaffolding and the building blocks where it was there pretty early
43:1743 minutes, 17 secondson and so the challenge is not any one scenario or situation for which you know we fail at 100% of those it's more you
43:2643 minutes, 26 secondsknow we're benchmarking against a pretty good or pretty high standard which is human driving all things considered humans are excellent at handling the
43:3443 minutes, 34 secondsedge cases and unexpected scenarios whereas computers the opposite and so beating that that baseline set by humans
43:4243 minutes, 42 secondsis the challenge and so what we've been doing for quite some time now is basically it's this continuous improvement process
43:5043 minutes, 50 secondswhere we find sort of the the most you know uncomfortable or the things that that could lead to a safety issue other
44:0044 minutesthings all these events and then we sort of categorize them and rework parts of our system to make incremental improvements and do that over and over
44:0744 minutes, 7 secondsand over again and we just see sort of the overall performance of the system you know actually increasing in a pretty steady clip but there's no one thing
44:1544 minutes, 15 secondsthere's actually like thousands of little things and just like polishing functionality and making sure that it handles you know every version
44:2244 minutes, 22 secondsimpossible permutation of a situation by either applying more deep learning systems or just by you know adding more
44:3244 minutes, 32 secondstests coverage or new scenarios that that we develop against and just grinding on that it's we're sort of in the the unsexy phase of development
44:3944 minutes, 39 secondsright now which is doing the real engineering work that it takes to go from prototype to production you're basically scaling the the
44:4544 minutes, 45 secondsgrinding so has sort of taking seriously that the process of all those edge cases both with human experts and machine
44:5444 minutes, 54 secondslearning methods to cover to cover all those situations yeah and the exciting thing for me is I don't think that grinding ever stops right because
45:0345 minutes, 3 secondsthere's a moment in time where you you cross that threshold of human performance and become superhuman but
45:1145 minutes, 11 secondsthere's no reason there's no first principles reason that AV capability will tap out anywhere near humans like there's no reason it couldn't be 20
45:1945 minutes, 19 secondstimes better whether that's you know just better driving or safer driving a more comfortable driving or even a thousand times better given enough time
45:2645 minutes, 26 secondsand we intend to basically chase that you know forever to build the best possible product better and better and
Chapter 25: Autonomous Vehicles
45:3345 minutes, 33 secondsbetter and always new educators come up and you experiences so and you want to automate that process as much as
45:3945 minutes, 39 secondspossible mhm so what do you think in general in society when do you think we may have hundreds of thousands of fully
45:4845 minutes, 48 secondsautonomous vehicles driving around so first of all predictions nobody knows the future you're a part of the leading people trying to define that future but
45:5645 minutes, 56 secondseven then you still don't know but if you think about a hundreds of thousands of heat so a significant fraction of vehicles in
46:0546 minutes, 5 secondsmajor cities are autonomous do you think I would Rodney Brooks who is 2050 and
46:1246 minutes, 12 secondsbeyond are you more with Elon Musk who is we should have had that two years ago
46:1946 minutes, 19 secondswell I mean I don't want me to have it two years ago but we're not there yet so I guess the the way I would think about
46:2846 minutes, 28 secondsthat is let's let's flip that question around so what would prevent you to reach hundreds of thousands of vehicles
46:3546 minutes, 35 secondsand that's a goodness a good rephrasing yeah so the
46:4146 minutes, 41 secondsI'd say the it seems the consensus among the people developing self-driving cars
46:4946 minutes, 49 secondstoday is to sort of start with some form of an easier environment whether it means you know lacking inclement weather or you know mostly sunny or whatever it
46:5846 minutes, 58 secondsis and then add add capability for more complex situations over time and so if
47:0547 minutes, 5 secondsyou're only able to deploy in areas that that meet sort of your criteria or that the current domain you know operating
47:1347 minutes, 13 secondsdomain of the software you developed that may put a cap on how many cities you could deploy in but then as those restrictions start to
47:2047 minutes, 20 secondsfall away like maybe you add you know capability to drive really well and and safely in heavy rain or snow you know that that probably opens up the market
47:2947 minutes, 29 secondsby - two or three fold in terms of the cities you can expand into and so on and so the real question is you know I I know today if we wanted to we could
47:3747 minutes, 37 secondsproduce that that many autonomous vehicles but we wouldn't be able to make use of all of them yet because we would sort of saturate the demand in the
47:4547 minutes, 45 secondscities in which we would want to operate initially so if I were to guess like what the timeline is for those things falling away and reaching hundreds of
47:5247 minutes, 52 secondsthousands of vehicles maybe a range is but I would I would say less than five years that's in five years yeah and of
Chapter 26: Building a Successful Startup
48:0048 minutescourse you're working hard to make that happen so you started two companies that were eventually acquired for each for a
48:0748 minutes, 7 secondsbillion dollars so you're pretty good person to ask what does it take to build a successful startup mmm-hmm I think
48:1548 minutes, 15 secondsthere's there sort of survivor bias here a little bit but I can try to find some common threads for the the things that worked for me which is
48:2448 minutes, 24 secondsyou know in in both of these companies it was really passionate about the core technology I actually like you know lay awake at night thinking about these problems and how to solve them and I
48:3348 minutes, 33 secondsthink that's helpful because when you start a business there are like to this day they're they're these crazy ups and downs like one day you think the
48:4148 minutes, 41 secondsbusiness is just on you're just on top of the world and unstoppable and the next day you think okay this is all gonna and you know it's it's just it's just going south and it's gonna be over
48:4848 minutes, 48 secondstomorrow and and so I think like having a true passion that you can fall back on and knowing that you would be doing it
48:5648 minutes, 56 secondseven if you weren't getting paid for it helps you whether those those tough times so that's one thing I think the other one is
49:0449 minutes, 4 secondsreally good people so I've always been surrounded by really good co-founders that are logical thinkers are always pushing their limits and have very high
49:1249 minutes, 12 secondslevels of integrity so that's Dan Khan in my current company and actually his brother and a couple other guys for Justin TV and twitch and then I think
49:2049 minutes, 20 secondsthe last thing is just uh I guess persistence or perseverance like and and that that can
49:2749 minutes, 27 secondsapply to sticking to sort of a or having conviction around the original premise of your idea and and sticking around to
49:3449 minutes, 34 secondsdo all the you know the unsexy work to actually make it come to fruition including dealing with you know whatever it is that that you're not passionate
49:4349 minutes, 43 secondsabout whether that's finance or or HR or or operations or those things as long as you are grinding away in working towards
49:5149 minutes, 51 secondsyou know that North Star for your business whatever it is and you don't give up and you're making progress every day it seems like eventually you'll end up in a good place and the only things
49:5949 minutes, 59 secondsthat can slow you down are you know running out of money or I suppose your competitors destroying you but I think most of the time it's people giving up or or somehow destroying things
50:0750 minutes, 7 secondsthemselves rather than being beaten by their competition or running out of money yeah if you never quit eventually you'll arrive so working size version of what I was
Chapter 27: Y Combinator vs VC Route
50:1650 minutes, 16 secondstrying to say yeah so you want the Y Combinator out twice yeah what do you think in a quick question do you think
50:2450 minutes, 24 secondsis the best way to raise funds in the early days or not just funds but just community develop your idea and so on
50:3150 minutes, 31 secondscan you do it solo or maybe with a co-founder with like self-funded do you
50:3950 minutes, 39 secondsthink Y Combinator is good it's good to do VC route is there no right answer was there for the Y Combinator experience something that you could take away that
50:4750 minutes, 47 secondsthat was the right path to take there's no one-size-fits-all answer but if your ambition I think is to you know see how big you can make something or or or
50:5550 minutes, 55 secondsrapidly expand and capture market or solve a problem or whatever it is then then you know going to venture back route is probably a good approach so
51:0451 minutes, 4 secondsthat so that capital doesn't become your primary constraint Y Combinator I love because it puts you in this sort of
51:1251 minutes, 12 secondscompetitive environment while you're where you're surrounded by you know the top maybe one percent of other really highly motivated you know peers who are
51:2051 minutes, 20 secondsin the same same place and that that environment I think just breeds breed success right if you're surrounded by really brilliant hard-working people
51:2851 minutes, 28 secondsyou're gonna feel you know sort of compelled or inspired to try to emulate them and/or beat them and so even though
51:3651 minutes, 36 secondsI had done it once before and I felt like yeah I'm pretty self-motivated I thought like I look this is gonna be a hard problem I can use all the help I
51:4351 minutes, 43 secondscan get so surrounding myself with other entrepreneurs is gonna make me work a little bit harder or push a little harder than it's worth it when Saba
51:5151 minutes, 51 secondswhite why I did it you know for example a second time let's let's go philosophical existential if you'd go back and do something differently in
Chapter 28: Philosophical existential
51:5851 minutes, 58 secondsyour life starting in high school than MIT leaving MIT you could have gone the
52:0652 minutes, 6 secondsPG route doing startup I'm gonna see about a start-up in California and youth or maybe some aspects of fundraising is
52:1552 minutes, 15 secondsthere something you'll regret something you need not necessarily grab but if you go back it could do differently I think I've made a lot of
52:2252 minutes, 22 secondsmistakes like you know pretty much everything you can screw up I think I've screwed up at least once but I you know I don't regret those things I think it's
52:3152 minutes, 31 secondshard to hard to look back on things even if they didn't go well and call it a regret because hopefully took away some new knowledge or learning from that so
52:4252 minutes, 42 secondsI would say there was a period yeah the closest I can I can come to us is there's a period in in justin.tv I think
52:4952 minutes, 49 secondsafter seven years where that the company was going one direction which is sorts twitch in video gaming
52:5752 minutes, 57 secondsand I'm not a video gamer I don't really even use twitch at all and I was still working on the core technology there but
53:0453 minutes, 4 secondsmy heart was no longer in it because the business that we were creating was not something that I was personally passionate about it didn't meet your bar of existential impact yeah and I'd say
53:1353 minutes, 13 secondsIII probably spent an extra year or two working on that and and I'd say like I would have just tried to do something
53:2053 minutes, 20 secondsdifferent sooner because those are those were two years where I felt like you know from this philosophical or
53:2853 minutes, 28 secondsexistential thing I I just I just felt something was missing and so I would have I would have if I could look back now and tell myself it's like I would have said exactly that like you're not
53:3653 minutes, 36 secondsgetting any meaning out of your work personally right now you should you should find a way to change that and that's part of the pitch I use to
53:4453 minutes, 44 secondsbasically everyone who joins crews today it's like hey you've got that now by coming here well maybe you needed the two years of that existential dread to develop the
Chapter 29: What does 2019 hold for Crew
53:5153 minutes, 51 secondsfeeling that ultimately was the fire that created crews so you never know you can be good theory yeah so last question
53:5853 minutes, 58 secondswhat does 2019 hold for crews after this I guess we're gonna go and I'll talk to your class but one of the big things is
54:0554 minutes, 5 secondsgoing from prototype to production for autonomous cars and what does that mean once that look like in 2019 for us is the year that we try to cross over that
54:1454 minutes, 14 secondsthreshold and reach you know superhuman level of performance to some degree with the software and have all the other of the thousands of
54:2154 minutes, 21 secondslittle building blocks in place to launch you know our first commercial product so that's that's what's in score for us are in store for us and we've got
54:3054 minutes, 30 secondsa lot of work to do we've got a lot of brilliant people working on it so it's it's all up to us now yeah from Charlie
54:3954 minutes, 39 secondsMiller and Chris fells like the people I have crossed paths with if you know it sounds like you have an amazing team so I'm like I said it's one of the most I
54:4854 minutes, 48 secondsthink one of the most important problems in artificial intelligence of the century you'll be one of the most defining the super exciting that you
54:5454 minutes, 54 secondswork on it and the best of luck in 2019 I'm really excited to see what Cruz comes up with thank you thanks for having me today
55:02