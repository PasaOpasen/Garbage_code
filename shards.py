
import math




class Shard:

    def __init__(self, left, right):

        self.left = left
        self.right = right

        self.pressure = 0
        self.pressure_time = 0
        self.stable = False

    def update_pressure(self, left_border, right_border):

        if any( (self.left <= left_border <=  self.right , self.left <= right_border <=  self.right )): # если хотя бы одна из границ приходится на этот шард, +1 к нагрузке

            self.pressure += 1
    
    def update_time(self, new_shard_array):

        if self.pressure > 100: 
            self.pressure_time += 1
            self.stable = False
        else: 
            self.pressure_time = 0
            self.stable = True

        if self.pressure_time == 60*5:

            new_shard_array.append(Shard(self.left, math.ceil((self.left + self.right)/2)))

            self.left = math.ceil((self.left + self.right)/2) + 1

            self.pressure_time = 0
            self.stable = True
        
        self.pressure = 0



NOW = 0
shards = [Shard(0, 1)]

while NOW < 2500:
    
    # print(f"NOW = {NOW}")
    for N in range(300):

        if N < math.floor(NOW / 10):
            a, b = NOW - (N+1)*10, NOW - N*10
        else:
            a, b = max(0, NOW - 10), NOW
        
        # print(a, b)
        
        for s in shards: s.update_pressure(a, b)
    
    for s in shards:
        s.update_time(shards)
    
    NOW += 1

    shards[0].right = NOW + 1
    
    if all((s.stable for s in shards)): print(NOW/60)

    #print()


print(NOW/60)
















