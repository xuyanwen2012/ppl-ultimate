
target("ppl")
    set_kind("static")
    add_includedirs("$(projectdir)/include")
    add_files("host/*.cpp")
    add_packages("glm")
    add_cxxflags("-pthread")

target("ppl-cuda")
    set_kind("static")
    add_includedirs("$(projectdir)/include")
    add_files("cuda/*.cu")
    add_packages("glm")
    add_cugencodes("native")
