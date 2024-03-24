
target("ppl")
    set_kind("static")
    add_includedirs("$(projectdir)/include")
    add_files("host/*.cpp")
    add_packages("glm", "pthread")

target("ppl-hybrid")
    set_kind("static")
    add_includedirs("$(projectdir)/include")
    add_files(
        "cuda/*.cu",
        "host/sort.cpp"
        -- Must exclude "host/structures.cpp"
        )
    add_packages("glm")
    add_cugencodes("native")
