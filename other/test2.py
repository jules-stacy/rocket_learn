





#launch bakkesmod - single instance
import subprocess
bakkesmod = "C:\\Program Files\\BakkesMod\\BakkesMod.exe"
subprocess.call([bakkesmod])

#launch a duel environment
#Currently launches then crashes. Missing agent and reward function.
import rlgym
env = rlgym.make("Duel")


#monitor system resources
import psutil
psutil.cpu_percent() #check cpu usage
psutil.virtual_memory()[2] #check RAM percentage
