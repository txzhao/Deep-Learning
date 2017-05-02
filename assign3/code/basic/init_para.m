function [W, b] = init_para(m, std)

for i = 1 : size(m, 2) - 1
    W{i} = std*rand(m(i + 1), m(i));
    b{i} = zeros(m(i + 1), 1);
end

end