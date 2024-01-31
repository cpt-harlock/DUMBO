# Structure of simulation code

In simulation_*.rs files you can find code use to instantiate simulation components and run the simulation.
The first part of the code instantiates simulation components as per the parameters specified from the command line.
After, these components are linked together to build the architecture of the system pipeline.

As an example:

```rust
    /* ARCHITECTURE BUILDING */
    parser.borrow_mut().set_next_block(hash_table.clone());
    /* NO BF */
    hash_table.borrow_mut().set_next_block(flow_manager.clone());
    //hash_table.borrow_mut().set_next_block(model_wrapper.clone());
    flow_manager.borrow_mut().set_model_next_block(model_wrapper.clone());
    flow_manager.borrow_mut().set_evicted_next_block(cms.clone());
    model_wrapper.borrow_mut().set_mice_next_block(cms.clone());
    model_wrapper.borrow_mut().set_hh_next_block(hash_table.clone());
    cms.borrow_mut().set_control_plane_block(control_plane.clone());
```

What it is happening is straightforward: for each object, we set the binding to the next one or more component (by saving a clone of the object reference).
For example, the parser has only output block, which in this case is the hash table. The flow manager hash instead to outputs, one for the flows to be classified and one for flows evicted before collecting required amount of packets.
The saved references refer to an object that implements PacketReceiver interface (explained later): any object implementing this interface can be attached.

Then the code just iterates over the list of packets parsed by the parser object. On each packet, we invoke the method "receive_packet" of the block chained to the parser. The method return an Option< reference-to-packet-receiver >: if it's != None, we invoke the receive_packet on this next block, otherwise the simulation for the packet is terminated.
