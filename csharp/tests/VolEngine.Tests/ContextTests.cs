namespace VolEngine.Tests;

public sealed class ContextTests
{
    [Fact]
    public void Default_threads_is_the_hardware_concurrency_capped_at_1024()
    {
        using var context = new Context();
        Assert.Equal(Math.Min(Environment.ProcessorCount, 1024), context.Threads);
    }

    [Fact]
    public void Threads_can_be_set()
    {
        using var context = new Context(2);
        Assert.Equal(2, context.Threads);
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(1025)]
    public void Threads_outside_0_to_1024_are_rejected(int threads)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Context(threads));
    }

    [Fact]
    public void A_disposed_context_throws()
    {
        var context = new Context();
        context.Dispose();
        context.Dispose();
        Assert.Throws<ObjectDisposedException>(() => context.Threads);
    }
}
