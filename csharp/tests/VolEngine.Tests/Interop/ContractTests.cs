using VolEngine.Tests.Support;

namespace VolEngine.Tests;

public sealed class ContractTests
{
    [Fact]
    public void Every_contract_function_has_a_matching_declaration()
    {
        Assert.Empty(ContractCheck.FunctionProblems(Repository.ReadContract(), typeof(NativeMethods)));
    }

    [Fact]
    public void Every_contract_define_matches_a_member()
    {
        Assert.Empty(ContractCheck.DefineProblems(Repository.ReadContract()));
    }

    [Fact]
    public void A_missing_declaration_is_reported()
    {
        var problems = ContractCheck.FunctionProblems(["function ve_status ve_not_declared(int32_t)"], typeof(NativeMethods));
        Assert.Contains(problems, problem => problem.Contains("ve_not_declared", StringComparison.Ordinal));
    }

    [Fact]
    public void A_wrong_parameter_type_is_reported()
    {
        var line = "function ve_status ve_context_threads(const ve_context*, double*)";
        var problems = ContractCheck.FunctionProblems([line], typeof(NativeMethods));
        Assert.Contains(problems, problem => problem.Contains("ve_context_threads", StringComparison.Ordinal));
    }

    [Fact]
    public void A_declaration_outside_the_contract_is_reported()
    {
        var problems = ContractCheck.FunctionProblems(["function int32_t ve_abi_version(void)"], typeof(NativeMethods));
        Assert.Contains(problems, problem => problem.Contains("ve_fit_batch", StringComparison.Ordinal));
    }

    [Fact]
    public void An_unmapped_define_is_reported()
    {
        var problems = ContractCheck.DefineProblems(["define VE_NEW_THING 3"]);
        Assert.Contains(problems, problem => problem.Contains("VE_NEW_THING", StringComparison.Ordinal));
    }

    [Fact]
    public void The_real_panel_ranges_declaration_is_accepted()
    {
        var line = "function ve_status ve_panel_set_ranges(ve_panel*, const size_t*, const size_t*)";
        var problems = ContractCheck.FunctionProblems([line], typeof(NativeMethods));
        Assert.DoesNotContain(problems, problem => problem.StartsWith("ve_panel_set_ranges ", StringComparison.Ordinal));
    }

    [Fact]
    public void A_const_input_declared_as_out_is_reported()
    {
        AssertWrong("function ve_status ve_panel_set_ranges(ve_panel*, const size_t*, const size_t*)",
                    "ve_panel_set_ranges parameter 1");
    }

    [Fact]
    public void A_ref_output_is_reported()
    {
        AssertWrong("function ve_status ve_context_threads(const ve_context*, int32_t*)", "ve_context_threads parameter 1");
    }

    [Fact]
    public void A_handle_array_declared_as_out_is_reported()
    {
        AssertWrong("function ve_status ve_batch_set_specs(ve_batch*, const ve_spec* const*, size_t)",
                    "ve_batch_set_specs parameter 1");
    }

    [Fact]
    public void A_string_without_utf8_is_reported()
    {
        AssertWrong("function ve_status ve_spec_set_fixed(ve_spec*, const char*, double)", "ve_spec_set_fixed strings");
    }

    private static void AssertWrong(string line, string expected)
    {
        var problems = ContractCheck.FunctionProblems([line], typeof(WrongDeclarations));
        Assert.Contains(problems, problem => problem.StartsWith(expected, StringComparison.Ordinal));
    }
}
