# scope

### a memory ownership model for gc free data driven pipelines.

## model

the `libevt` c implementation relies on a strict memory ownership model to manage system resources, over implementing a garbage collection strategy.

**NOTE this is strictly for heap managed resources**

this approach provides some immediate benefits:

  1. predictable latency across the pipeline
  2. deterministic resource usage with clear ownership/scoping boundaries

with this in mind, the pipeline needs to be incredibly strict in how it handles malloc/free. to achieve this, it approaches memory management in terms of scope. each contract in the protocol header files is essentially a self contained unit, which can build up smaller and smaller pieces. all types implement the following lifecycles  (the document will use *component* to reference type instance going forward):
```c
typedef struct Component Component;
// lifecycle initialize method, _create lifecyle
Component *component_create(<your-constructor-params>);
// lifecycle initialize method, _destroy lifecyle
void component__destroy(Component *c);
```

imagine that the system memory is really just a finite state space with a set number N segments evenly spaced, each representing a memory address, where a given scope can at any given time allocate up to M addresses, where M < N and M is the pool of available addresses. by introducing lifecycles to every container, we essentially create a contract with the system stating:

*when a component in the system is instantiated, it enters into a contract with the system that for the time that it is needed, it will reserve resources only until it completes its task or fails, where it will then clean up all of the resources associated with itself and its dependents*

let's also define a couple simple rules for the components:

1. a component can release anything back to system that is allocated within its scope once it is done using it
2. a component can never release anything that is not in its scope and it doesn't own

in this way, the memory ownership ends up being laid out like a tree, where the root is the top level scope and all dependencies point back up to the top level.

for instance, given the folowing types:
```c
typedef struct Resource
{
  uint64_t id;
} Resource;

typedef struct Component
{
  Resource *r;
} Component;
```

where each lifecycle implementation for `Resource` would be:
```c
Resource *resource_create(uint64_t id)
{
  // enter into the contract to borrow the address for the resource
  Resource *r = malloc(sizeof(Resource));
  if (!r)
    return NULL;
  r->id = id;
  return r;
}

void resource_destroy(Resource *r)
{
  if (!r)
    return;
  // the type has completed its lifecycle
  // it fulfills the contract by returning the memory back to the system
  free(r);
}
```

below compares a couple implementations of `Component` to see how it will handle the resources depending on what scope it came from.

1. instantiate the resource in a higher scope and pass into the constructor

the container is not the owner of resource, which is owned by a higher scope. here, the container is allowed to read and modify the value of the resource, but is not allowed to free the resource from memory. the owner of the resource will free the memory once it has completed its lifecycle. 
```c
Component *component_create(Resource *r)
{
  if (!r)
    return NULL;
  Component *c = malloc(sizeof(Component));
  if (!c)
    return NULL;
  c->r = r; // reference the pointer here
  return c;
}

void component_destroy(Component *c)
{
  if (!c)
    return;
  free(c);
}
```

2. instantiate the resource in the current scope in the constructor

the current scope is the owner of the resource and expects the resource's lifecycle to match its own. the current scope then handles the lifecycle of the resource and releases it once its lifecycle completes.
```c
Component *component_create(uint64_t r_id)
{
  Component *c = malloc(sizeof(Component));
  if (!c)
    return NULL;
  // the current scope is the owner of the resource
  Resource *r = resource_create(r_id);
  if (!r) {
    free(c); // failed to allocate dependencies so will release to system
    return NULL;
  }
  c->r = r;
  return c;
}

void component_destroy(Component *c)
{
  if (!c)
    return;
  // clean up the resource since it is a dependent
  // part of the contract is freeing everything that the current scope allocated
  resource_destroy(c->r);
  free(c);
}
```

3. instantiate on demand in functions

this can be considered request scoped, so the lifecycle of the resource is tied to the scope of the function call. once the function completes, allocated the resource is released.
```c
// here type doesn't embed the resource, for clarity in the example
typedef struct Component Component;

// just some method instance on Component
void component_operation(Component *c, uint64_t r_id)
{
  if (!t)
    return;
  // scoped to the function
  Resource *r = resource_create(r_id);
  /* ... do something ... */
  // clean up now
  resource_destroy(r);
}
```

**NOTE copying pointers does not share ownership**

to avoid the following error, strictly release the *original resource* not the reference.

```c
Component *c = component_create(r);
// shallow copy/reference to c
Component *c2 = c;
component_destroy(c);
// this will result in double free error if uncommented
// component_destroy(c);
```

## application

with the above in mind, memory management becomes deterministic. every component in the system knows exactly how to create and destroy itself once its context/task has completed or failed. it becomes straightforward to reason about the chain of resource allocations/deallocations.

from this approach, utilizing scopes and explicit lifecycles, naturally emerges a framework for event-driven asynchronous systems. concurrent operations are each individually scoped and each follow the rules laid out above, where each manages its own lifecycle and the lifecycle of its depedency tree. global state can still be passed around but is managed by the calling process, not by individual operations. taking this a step further, each asynchronous operation can essentially be boiled down to a containing type, defined as such:
```c
/*
  data -> the scope of the callback
  op -> the callback fn to call once the context is no longer deferred
*/
typedef struct Context
{
  void *data;
  callback_operator op;
} Context;

Context *context_create(void *data, callback_operator op);
void context_destroy(Context *ctx);

/*
context_resolve
  essentially:
    ctx->op(ctx->data);
*/
void context_resolve(Context *ctx);
```

the context is a runtime resource with the only task being to provide a way to defer the execution of an operation. when a context is created, it binds the reference to data from the scope of the caller. once `context_resolve` is executed, the callback has been completed, and the context enters its destroy lifecycle. since the data is from a higher scope (as it is a dependency in the constructor), it is not freed but instead returned back to the caller. this creates almost an envelope that unwraps when the system activates it.
