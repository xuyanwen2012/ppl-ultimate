add_rules("mode.debug", "mode.release")


set_languages("cxx17")
set_warnings("all")

add_requires("glm")

if is_plat("linux") then
    add_requires("pthread")
end

includes("ppl")
includes("tests")
-- includes("demo")
