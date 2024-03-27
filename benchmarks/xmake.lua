add_requires("benchmark")

target("bench-cpu")
    set_kind("binary")
    add_includedirs("$(projectdir)/include")
    add_headerfiles("../include/**/*")
    add_files("*.cpp")
    add_packages("benchmark")
    add_deps("ppl")

target("bench-gpu")
    set_kind("binary")
    add_includedirs("$(projectdir)/include")
    add_headerfiles("../include/**/*")
    add_files("*.cu")
    add_cugencodes("native")
    add_packages("benchmark")
    add_deps("ppl-hybrid")
    