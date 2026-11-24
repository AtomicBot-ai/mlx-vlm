from functools import wraps

import mlx.core as mx

generation_stream = mx.new_thread_local_stream(mx.default_device())


def on_generation_stream(function):
    """Run every generator step and cleanup on the generation stream."""

    @wraps(function)
    def generate(*args, **kwargs):
        iterator = function(*args, **kwargs)
        try:
            while True:
                with mx.stream(generation_stream):
                    try:
                        value = next(iterator)
                    except StopIteration as done:
                        return done.value
                # Restore the caller's stream before returning control to it.
                yield value
        finally:
            with mx.stream(generation_stream):
                iterator.close()

    return generate
