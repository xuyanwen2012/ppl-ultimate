add_requires("benchmark")

target("bench-cpu")
    set_kind("binary")
    add_includedirs("$(projectdir)/include")
    add_files("*.cpp")
    add_packages("benchmark")
    add_deps("ppl")
    