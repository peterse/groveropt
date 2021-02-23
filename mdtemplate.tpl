%%writefile mytemplate.tpl

{% extends 'full.tpl'%}
{% block any_cell %}
    <div style="border:thin solid red">
        {{ super() }}
    </div>
{% endblock any_cell %}