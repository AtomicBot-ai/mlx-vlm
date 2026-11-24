import mlx.core as mx
import pytest

from mlx_vlm._generation_stream import generation_stream, on_generation_stream


@pytest.mark.parametrize("failure", [False, True])
def test_generation_stream_restores_caller_on_close_and_error(failure):
    seen = []

    @on_generation_stream
    def generate():
        try:
            seen.append(mx.default_stream(mx.default_device()))
            yield 1
            raise ValueError("generation failed")
        finally:
            seen.append(mx.default_stream(mx.default_device()))

    with mx.stream(generation_stream):
        expected = mx.default_stream(mx.default_device())
    caller = mx.new_stream(mx.default_device())
    with mx.stream(caller):
        iterator = generate()
        assert next(iterator) == 1
        assert mx.default_stream(mx.default_device()) == caller
        if failure:
            with pytest.raises(ValueError, match="generation failed"):
                next(iterator)
        else:
            iterator.close()
        assert mx.default_stream(mx.default_device()) == caller
    assert seen == [expected, expected]
