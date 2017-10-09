function [W, b] = init_para(m, std)

for i = 1 : size(m, 2) - 1
    if nargin < 2
        % Xavier initialization
        var = 2/(m(i) + m(i + 1));
        std = sqrt(var);
    end
    
    W{i} = std*rand(m(i + 1), m(i));
    b{i} = zeros(m(i + 1), 1);
end

end
