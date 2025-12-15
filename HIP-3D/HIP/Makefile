include ../config.mk
include flags.mk

all:	
	$(CC) $(CFLAGS) $(COMMON_FLAGS) -c hip_driver.cpp
	$(PGCC) $(PGCCFLAGS) $(COMMON_FLAGS) -c hip_stuff.cpp
	$(PGCC) $(PGCCFLAGS) $(COMMON_FLAGS) -c hip_propagate.cpp
	$(PGCC) $(PGCCFLAGS) $(COMMON_FLAGS) -c hip_insertsource.cpp

clean:
	rm -f *.o *.a

