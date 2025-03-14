---
marp: true
theme: gaia

---

![bg](https://e1.pxfuel.com/desktop-wallpaper/392/852/desktop-wallpaper-i-made-the-new-horizons-dock-design-into-a-more-decorated-animal-crossing.jpg)

# Simulating Animal Crossing New Horizons

#### MSDS460 Decision Analytics - Term Project  
#### Sally Lee

---
![bg](https://wallpapers.com/images/hd/aesthetic-landscape-animal-crossing-hd-xx4gl44cuw3ppkmm.jpg)


## Problem Definition

- **Objective:** Simulate the dynamic behaviors of autonomous entities in a virtual environment
- **Goal:** Understand emergent behaviors and social interactions in life-simulation games
- **Methodology:** Python’s mesa module to create an agent-based simulation

---
![bg](https://wallpapers.com/images/hd/aesthetic-landscape-animal-crossing-hd-xx4gl44cuw3ppkmm.jpg)
## Applications and Literature

- Agent-based modeling (ABM) has been widely used in pandemic modeling like the recent COVID-19 spread
- ABM helps urban planners study traffic congestion and autonomous vehicle
-  Foundational work by Thomas Schelling Dynamic Models of Segregation 1971


---
![bg](https://wallpapers.com/images/hd/aesthetic-landscape-animal-crossing-hd-xx4gl44cuw3ppkmm.jpg)
## Design

- Each villager is an autonomous agent with a randomly selected predefined 
personality influencing activities and sleep cycles
- Simulation iterates through daily cycles, tracking movement, interactions, and tasks
- Used publicly available game data for parameters and states



---

![bg](https://wallpapers.com/images/hd/aesthetic-landscape-animal-crossing-hd-xx4gl44cuw3ppkmm.jpg)
### Future Improvements
- **Path-finding:** Navigation using algorithms e.g. Dijkstra’s 
- **Advanced decision-making:** Reacting to environment and events
- **Weather system:** Influences behavior
- **Emotions/Mood:** Agent interactions can affect mood and influence behavior