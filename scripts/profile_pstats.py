import pstats
p = pstats.Stats('profile.out')
p.sort_stats('cumulative').print_stats(20)  # Top 10 by cumulative time
