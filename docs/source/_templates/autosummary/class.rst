{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:

   {% block methods %}
   {% set shown = methods | reject("equalto", "__init__") | list %}
   {% if shown %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in shown %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
